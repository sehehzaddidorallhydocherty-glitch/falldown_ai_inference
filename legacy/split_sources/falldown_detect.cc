/* Copyright (c) 2023, Canaan Bright Sight Co., Ltd
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.f
 */

#include "falldown_detect.h"
#include "utils.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

// for image
falldownDetect::falldownDetect(const char *kmodel_file, float obj_thresh,float nms_thresh,  const int debug_mode): obj_thresh_(obj_thresh),nms_thresh_(nms_thresh), AIBase(kmodel_file,"falldownDetect", debug_mode)
{

    model_name_ = "falldownDetect";
    
    ai2d_out_tensor_ = get_input_tensor(0);
}   

// for video
falldownDetect::falldownDetect(const char *kmodel_file, float obj_thresh,float nms_thresh, FrameCHWSize isp_shape, uintptr_t vaddr, uintptr_t paddr, const int debug_mode): obj_thresh_(obj_thresh),nms_thresh_(nms_thresh), AIBase(kmodel_file,"falldownDetect", debug_mode)
{
    model_name_ = "falldownDetect";
    
    vaddr_ = vaddr;

    isp_shape_ = isp_shape;
    dims_t in_shape{1, isp_shape_.channel, isp_shape_.height, isp_shape_.width};
    int isp_size = isp_shape_.channel * isp_shape_.height * isp_shape_.width;
    #if 0
    ai2d_in_tensor_ = host_runtime_tensor::create(typecode_t::dt_uint8, in_shape, { (gsl::byte *)vaddr, isp_size },
        true, hrt::pool_shared).expect("cannot create input tensor");
    #else
    ai2d_in_tensor_ = hrt::create(typecode_t::dt_uint8, in_shape, hrt::pool_shared).expect("create ai2d input tensor failed");
    #endif

    // ai2d_out_tensor
    ai2d_out_tensor_ = get_input_tensor(0);
    // fixed padding resize param
    Utils::padding_resize(isp_shape_, {input_shapes_[0][3], input_shapes_[0][2]}, ai2d_builder_, ai2d_in_tensor_, ai2d_out_tensor_, cv::Scalar(114, 114, 114));
}

falldownDetect::~falldownDetect()
{

}

void falldownDetect::set_frame(uintptr_t vaddr, uintptr_t paddr)
{
    vaddr_ = vaddr;
    (void)paddr;
}

// ai2d for image
void falldownDetect::pre_process(cv::Mat ori_img)
{
    ScopedTiming st(model_name_ + " pre_process image", debug_mode_);
    std::vector<uint8_t> chw_vec;
    Utils::hwc_to_chw(ori_img, chw_vec);
    Utils::padding_resize({ori_img.channels(), ori_img.rows, ori_img.cols}, chw_vec, {input_shapes_[0][3], input_shapes_[0][2]}, ai2d_out_tensor_, cv::Scalar(114, 114, 114));
}

// ai2d for video
void falldownDetect::pre_process()
{
    ScopedTiming st(model_name_ + " pre_process video", debug_mode_);
    #if 0
    ai2d_builder_->invoke().expect("error occurred in ai2d running");
    #else
    size_t isp_size = isp_shape_.channel * isp_shape_.height * isp_shape_.width;
    if (vaddr_ == 0 || isp_size == 0)
    {
        std::printf("[ai] invalid video frame: vaddr=%p size=%zu\n", reinterpret_cast<void *>(vaddr_), isp_size);
        return;
    }
    auto mapped_buf = ai2d_in_tensor_.impl()->to_host().unwrap()->buffer().as_host().unwrap().map(map_access_::map_write).unwrap();
    auto buf = mapped_buf.buffer();
    memcpy(reinterpret_cast<char *>(buf.data()), (void *)vaddr_, isp_size);
    hrt::sync(ai2d_in_tensor_, sync_op_t::sync_write_back, true).expect("sync write_back failed");
    ai2d_builder_->invoke(ai2d_in_tensor_,ai2d_out_tensor_).expect("error occurred in ai2d running");
    // run ai2d
    #endif
    
}

void falldownDetect::inference()
{
    this->run();
    this->get_output();
}

void falldownDetect::post_process(FrameSize frame_size,std::vector<BoxInfo> &result)
{
    if (p_outputs_.empty() || output_shapes_.empty())
    {
        std::printf("[ai] no model output for post process\n");
        return;
    }

    const int net_h = input_shapes_[0][2];
    const int net_w = input_shapes_[0][3];

    if (p_outputs_.size() == 1)
    {
        const auto &shape = output_shapes_[0];
        if (shape.size() < 3)
        {
            std::printf("[ai] unsupported single output rank: %zu\n", shape.size());
            return;
        }

        const bool chw_layout = shape[1] >= 6 && shape[1] <= 256;
        const bool hwc_layout = shape[2] >= 6 && shape[2] <= 256;
        if (!chw_layout && !hwc_layout)
        {
            std::printf("[ai] unsupported single output shape: %dx%dx%d\n", shape[0], shape[1], shape[2]);
            return;
        }

        const int channels = chw_layout ? shape[1] : shape[2];
        const int num_boxes = chw_layout ? shape[2] : shape[1];
        if (channels < 6 || num_boxes <= 0)
        {
            std::printf("[ai] invalid single output shape: channels=%d boxes=%d\n", channels, num_boxes);
            return;
        }

        float *output = p_outputs_[0];
        auto value_at = [&](int c, int i) -> float {
            return chw_layout ? output[c * num_boxes + i] : output[i * channels + c];
        };

        const float gain = std::min(static_cast<float>(net_w) / frame_size.width,
                                    static_cast<float>(net_h) / frame_size.height);
        const float pad_x = (net_w - frame_size.width * gain) * 0.5f;
        const float pad_y = (net_h - frame_size.height * gain) * 0.5f;

        for (int i = 0; i < num_boxes; ++i)
        {
            int label = 0;
            float score = value_at(4, i);
            for (int cls = 1; cls < channels - 4; ++cls)
            {
                float cls_score = value_at(4 + cls, i);
                if (cls_score > score)
                {
                    score = cls_score;
                    label = cls;
                }
            }

            if (score <= obj_thresh_)
            {
                continue;
            }

            const float cx = value_at(0, i);
            const float cy = value_at(1, i);
            const float w = value_at(2, i);
            const float h = value_at(3, i);

            float x1 = (cx - w * 0.5f - pad_x) / gain;
            float y1 = (cy - h * 0.5f - pad_y) / gain;
            float x2 = (cx + w * 0.5f - pad_x) / gain;
            float y2 = (cy + h * 0.5f - pad_y) / gain;

            BoxInfo box;
            box.x1 = std::max(0.0f, std::min(static_cast<float>(frame_size.width), x1));
            box.y1 = std::max(0.0f, std::min(static_cast<float>(frame_size.height), y1));
            box.x2 = std::max(0.0f, std::min(static_cast<float>(frame_size.width), x2));
            box.y2 = std::max(0.0f, std::min(static_cast<float>(frame_size.height), y2));
            box.score = score;
            box.label = label;

            if (box.x1 < box.x2 && box.y1 < box.y2)
            {
                result.push_back(box);
            }
        }

        Utils::nms(result, nms_thresh_);
        return;
    }

    if (p_outputs_.size() < 3)
    {
        std::printf("[ai] unsupported output count: %zu\n", p_outputs_.size());
        return;
    }

    int net_len = net_h;
    // first output
    {
        float *output_0 = p_outputs_[0];
        auto boxes0 = Utils::decode_infer(output_0, net_len, 8, classes_num_, frame_size, anchors_0_, obj_thresh_);
        result.insert(result.begin(), boxes0.begin(), boxes0.end());
    }

    // second output
    {
        float *output_1 = p_outputs_[1];

        auto boxes1 = Utils::decode_infer(output_1, net_len, 16, classes_num_, frame_size, anchors_1_, obj_thresh_);
        result.insert(result.begin(), boxes1.begin(), boxes1.end());
    }
    
    // third output
    {
        float *output_2 = p_outputs_[2];

        auto boxes2 = Utils::decode_infer(output_2, net_len, 32, classes_num_, frame_size, anchors_2_, obj_thresh_);
        result.insert(result.begin(), boxes2.begin(), boxes2.end());
    }

    Utils::nms(result, nms_thresh_);
}
