# ONNX Windows 测试工程

本目录包含用于测试 `models/best_fp16.onnx` 的 Windows 侧脚本。脚本可以对图片进行批量推理，保存可视化预测结果，并输出简单的预测统计信息。由于当前流程不读取标注文件，因此不会计算 mAP、Precision、Recall 等真实精度指标。

## 安装依赖

在本目录下运行：

```powershell
py -m pip install -r requirements.txt
```

检查环境：

```powershell
py -c "import onnxruntime, numpy, cv2"
```

## 运行

示例：

```powershell
py eval_onnx_windows.py --model ..\..\models\best_fp16.onnx --images <image_dir> --out outputs --conf 0.5 --iou 0.30
```

参数说明：

- `--model`：ONNX 模型路径。默认值：`..\..\models\best_fp16.onnx`
- `--images`：测试图片所在文件夹。
- `--out`：输出文件夹。默认值：`outputs`
- `--conf`：置信度阈值。默认值：`0.5`
- `--iou`：NMS IoU 阈值。默认值：`0.30`

## 输出内容

- `predictions.csv`：每个检测框对应一行。
- `summary.json`：图片数量、检测数量、类别数量和置信度统计。
- `vis/`：绘制了预测框的图片。

如果 ONNX 元数据可用，类别信息会从模型中读取。当前 `best_fp16.onnx` 的元数据为：

- `0`: `Normal`
- `1`: `Fall`

## 说明

模型输入期望为 `images [1, 3, 640, 640]`，输出期望为 `[1, 6, 8400]`。脚本使用 letterbox 缩放，并使用填充值 `114`，以匹配板端部署时的预处理方式。

如需进行真实精度评估，后续可以补充 ground-truth 标签，并扩展脚本计算 mAP、Precision、Recall 和 F1。
