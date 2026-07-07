import argparse
import ast
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


DEFAULT_CLASS_NAMES = ["Normal", "Fall"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Windows-side ONNX batch inference for falldown detection."
    )
    parser.add_argument("--model", default="../../models/best_fp16.onnx", help="Path to ONNX model.")
    parser.add_argument("--images", default="calib_images", help="Directory containing test images.")
    parser.add_argument("--out", default="outputs", help="Output directory.")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.30, help="NMS IoU threshold.")
    return parser.parse_args()


def list_images(image_dir):
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"--images must be a directory: {root}")

    paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(paths)


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    src_h, src_w = image.shape[:2]
    dst_w, dst_h = new_shape
    gain = min(dst_w / src_w, dst_h / src_h)
    new_w = int(round(src_w * gain))
    new_h = int(round(src_h * gain))

    pad_w = dst_w - new_w
    pad_h = dst_h - new_h
    left = int(round(pad_w / 2 - 0.1))
    right = int(round(pad_w / 2 + 0.1))
    top = int(round(pad_h / 2 - 0.1))
    bottom = int(round(pad_h / 2 + 0.1))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, gain, left, top


def preprocess(image, input_hw):
    input_h, input_w = input_hw
    padded, gain, pad_x, pad_y = letterbox(image, (input_w, input_h))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.transpose(2, 0, 1).astype(np.float32)
    tensor /= 255.0
    tensor = np.expand_dims(tensor, axis=0)
    return tensor, gain, pad_x, pad_y


def box_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_box = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = area_box + area_boxes - inter
    return inter / np.maximum(union, 1e-6)


def nms(boxes, scores, iou_thresh):
    if len(boxes) == 0:
        return []

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        current = order[0]
        keep.append(int(current))
        if order.size == 1:
            break
        ious = box_iou(boxes[current], boxes[order[1:]])
        order = order[1:][ious < iou_thresh]
    return keep


def xywh_to_xyxy(boxes):
    converted = np.empty_like(boxes)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return converted


def postprocess(
    output, image_shape, gain, pad_x, pad_y, conf_thresh, iou_thresh, class_names
):
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError(f"Unexpected ONNX output shape: {output.shape}")
    if output.shape[1] != 6:
        raise ValueError(f"Expected output shape [1, 6, N], got {output.shape}")

    predictions = output[0].T
    boxes = predictions[:, :4].astype(np.float32)
    scores_by_class = predictions[:, 4:].astype(np.float32)
    image_h, image_w = image_shape[:2]

    class_ids = np.argmax(scores_by_class, axis=1)
    scores = scores_by_class[np.arange(scores_by_class.shape[0]), class_ids]
    mask = scores >= conf_thresh
    if not np.any(mask):
        return []

    selected_boxes = xywh_to_xyxy(boxes[mask])
    selected_scores = scores[mask]
    selected_class_ids = class_ids[mask]

    selected_boxes[:, [0, 2]] = (selected_boxes[:, [0, 2]] - pad_x) / gain
    selected_boxes[:, [1, 3]] = (selected_boxes[:, [1, 3]] - pad_y) / gain
    selected_boxes[:, [0, 2]] = np.clip(selected_boxes[:, [0, 2]], 0, image_w)
    selected_boxes[:, [1, 3]] = np.clip(selected_boxes[:, [1, 3]], 0, image_h)

    valid = (selected_boxes[:, 2] > selected_boxes[:, 0]) & (
        selected_boxes[:, 3] > selected_boxes[:, 1]
    )
    selected_boxes = selected_boxes[valid]
    selected_scores = selected_scores[valid]
    selected_class_ids = selected_class_ids[valid]
    if len(selected_boxes) == 0:
        return []

    keep = nms(selected_boxes, selected_scores, iou_thresh)
    results = []
    for idx in keep:
        class_id = int(selected_class_ids[idx])
        x1, y1, x2, y2 = selected_boxes[idx]
        results.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id]
                if class_id < len(class_names)
                else f"class_{class_id}",
                "confidence": float(selected_scores[idx]),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
        )

    results.sort(key=lambda item: item["confidence"], reverse=True)
    return results


def draw_predictions(image, detections):
    canvas = image.copy()
    colors = {
        0: (0, 180, 0),
        1: (0, 0, 255),
    }
    for det in detections:
        class_id = det["class_id"]
        color = colors.get(class_id, (255, 0, 255))
        x1 = int(round(det["x1"]))
        y1 = int(round(det["y1"]))
        x2 = int(round(det["x2"]))
        y2 = int(round(det["y2"]))
        label = f'{det["class_name"]} {det["confidence"]:.2f}'

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y1, label_size[1] + baseline + 4)
        cv2.rectangle(
            canvas,
            (x1, label_y - label_size[1] - baseline - 4),
            (x1 + label_size[0] + 4, label_y),
            color,
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 2, label_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
    return canvas


def get_model_class_names(session):
    metadata = session.get_modelmeta().custom_metadata_map
    names_text = metadata.get("names")
    if not names_text:
        return DEFAULT_CLASS_NAMES

    try:
        names = ast.literal_eval(names_text)
    except (SyntaxError, ValueError):
        return DEFAULT_CLASS_NAMES

    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return DEFAULT_CLASS_NAMES


def create_session(model_path):
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1:
        raise ValueError(f"Expected one ONNX input, got {len(inputs)}")

    input_info = inputs[0]
    input_shape = input_info.shape
    if len(input_shape) != 4 or input_shape[1] != 3:
        raise ValueError(f"Expected input shape [1, 3, H, W], got {input_shape}")
    if input_shape[2] != 640 or input_shape[3] != 640:
        raise ValueError(f"Expected input shape [1, 3, 640, 640], got {input_shape}")

    if len(outputs) != 1:
        raise ValueError(f"Expected one ONNX output, got {len(outputs)}")
    class_names = get_model_class_names(session)
    return session, input_info.name, outputs[0].name, (input_shape[2], input_shape[3]), class_names


def write_predictions_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_summary(image_count, failed_images, rows, elapsed_sec, class_names):
    class_counts = {name: 0 for name in class_names}
    confidences = []
    for row in rows:
        class_counts[row["class_name"]] = class_counts.get(row["class_name"], 0) + 1
        confidences.append(float(row["confidence"]))

    if confidences:
        confidence_stats = {
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
            "mean": float(np.mean(confidences)),
            "median": float(np.median(confidences)),
        }
    else:
        confidence_stats = {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }

    return {
        "image_count": image_count,
        "failed_image_count": len(failed_images),
        "failed_images": failed_images,
        "detection_count": len(rows),
        "class_counts": class_counts,
        "confidence": confidence_stats,
        "elapsed_sec": elapsed_sec,
        "images_per_sec": image_count / elapsed_sec if elapsed_sec > 0 else None,
    }


def main():
    args = parse_args()
    model_path = Path(args.model)
    image_paths = list_images(args.images)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {args.images}")

    out_dir = Path(args.out)
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    session, input_name, output_name, input_hw, class_names = create_session(model_path)

    rows = []
    failed_images = []
    start = time.perf_counter()

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            failed_images.append(str(image_path))
            continue

        tensor, gain, pad_x, pad_y = preprocess(image, input_hw)
        output = session.run([output_name], {input_name: tensor})[0]
        detections = postprocess(
            output,
            image.shape,
            gain,
            pad_x,
            pad_y,
            args.conf,
            args.iou,
            class_names,
        )

        for det in detections:
            row = {"image": str(image_path)}
            row.update(det)
            rows.append(row)

        visualized = draw_predictions(image, detections)
        cv2.imwrite(str(vis_dir / image_path.name), visualized)

    elapsed = time.perf_counter() - start

    write_predictions_csv(out_dir / "predictions.csv", rows)
    summary = build_summary(len(image_paths), failed_images, rows, elapsed, class_names)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"Images: {len(image_paths)}")
    print(f"Detections: {len(rows)}")
    print(f"Output: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
