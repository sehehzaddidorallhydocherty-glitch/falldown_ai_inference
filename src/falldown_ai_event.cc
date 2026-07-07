#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

#include "ai_inference.h"
#include "fdd_ipc.h"

namespace
{
volatile sig_atomic_t g_stop = 0;

void handle_signal(int)
{
    g_stop = 1;
}

void print_usage(const char *name)
{
    std::cout << "Usage: " << name << " <kmodel> <debug_mode>" << std::endl
              << "For example:" << std::endl
              << "  ./" << name << " yolov5n-falldown.kmodel 0" << std::endl;
}

std::vector<FddIpcBox> to_ipc_boxes(const DetectResult &result)
{
    std::vector<FddIpcBox> boxes;
    boxes.reserve(result.boxes.size());
    for (const auto &box : result.boxes)
    {
        boxes.push_back({box.x1, box.y1, box.x2, box.y2});
        if (boxes.size() >= kFddMaxFallBoxes)
        {
            break;
        }
    }
    return boxes;
}
}

int main(int argc, char *argv[])
{
    std::cout << "falldown_ai_event built at " << __DATE__ << " " << __TIME__ << std::endl;
    if (argc != 3)
    {
        print_usage(argv[0]);
        return -1;
    }

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    const char *kmodel_path = argv[1];
    const int debug_mode = std::atoi(argv[2]);

    FddIpcContext ipc;
    std::string error;
    if (!fdd_ipc_open(ipc, true, &error))
    {
        std::cerr << "[ai-event] failed to initialize IPC: " << error << std::endl;
        return -1;
    }
    std::cout << "[ai-event] IPC ready; waiting for video frames" << std::endl;

    AiContext ai_ctx = {};
    bool ai_ready = false;
    uint32_t ai_width = 0;
    uint32_t ai_height = 0;
    uint32_t ai_channels = 0;
    uint64_t last_frame_seq = 0;

    while (!g_stop && !fdd_ipc_should_stop(ipc))
    {
        FddIpcFrameMeta meta;
        std::vector<uint8_t> frame;
        if (!fdd_ipc_wait_latest_frame(ipc, last_frame_seq, 100, meta, frame))
        {
            continue;
        }
        last_frame_seq = meta.seq;

        if (!ai_ready ||
            ai_width != meta.width ||
            ai_height != meta.height ||
            ai_channels != meta.channels)
        {
            if (ai_ready)
            {
                ai_deinit(&ai_ctx);
                ai_ready = false;
            }

            if (ai_init(&ai_ctx, kmodel_path, meta.channels, meta.height, meta.width) != 0)
            {
                std::cerr << "[ai-event] failed to initialize AI model" << std::endl;
                fdd_ipc_request_stop(ipc);
                fdd_ipc_close(ipc);
                fdd_ipc_unlink();
                return -1;
            }
            ai_width = meta.width;
            ai_height = meta.height;
            ai_channels = meta.channels;
            ai_ready = true;
            std::cout << "[ai-event] model initialized for input "
                      << ai_channels << "x" << ai_height << "x" << ai_width << std::endl;
        }

        if (frame.empty())
        {
            continue;
        }

        DetectResult result = ai_run_frame(&ai_ctx, reinterpret_cast<uintptr_t>(frame.data()), 0);
        std::vector<FddIpcBox> boxes = to_ipc_boxes(result);
        if (!fdd_ipc_publish_result(ipc, meta.seq, boxes, &error))
        {
            std::cerr << "[ai-event] failed to publish result: " << error << std::endl;
        }

        if (debug_mode > 1)
        {
            std::cout << "[ai-event] frame_seq=" << meta.seq
                      << " boxes=" << boxes.size() << std::endl;
        }
    }

    if (ai_ready)
    {
        ai_deinit(&ai_ctx);
    }
    fdd_ipc_close(ipc);
    fdd_ipc_unlink();
    std::cout << "[ai-event] stopped" << std::endl;
    return 0;
}
