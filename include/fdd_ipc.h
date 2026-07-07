#ifndef FDD_IPC_H
#define FDD_IPC_H

#include <cstddef>
#include <cstdint>
#include <semaphore.h>
#include <string>
#include <vector>

constexpr const char *kFddFrameShmName = "/fdd_fall_frame_shm";
constexpr const char *kFddResultShmName = "/fdd_fall_result_shm";
constexpr uint32_t kFddIpcVersion = 1;
constexpr uint32_t kFddFrameMagic = 0x46444446;  // FDDF
constexpr uint32_t kFddResultMagic = 0x46444452; // FDDR
constexpr int kFddFrameSlotCount = 3;
constexpr int kFddMaxFallBoxes = 32;
constexpr size_t kFddMaxFrameBytes = 1280 * 720 * 3;

struct FddIpcBox
{
    int32_t x1;
    int32_t y1;
    int32_t x2;
    int32_t y2;
};

struct FddIpcFrameMeta
{
    uint64_t seq = 0;
    uint64_t timestamp_us = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t channels = 0;
    uint32_t size = 0;
};

struct FddIpcDetectionResult
{
    uint64_t frame_seq = 0;
    std::vector<FddIpcBox> boxes;
};

struct FddIpcFrameSlot
{
    uint64_t seq;
    uint64_t timestamp_us;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t size;
    uint8_t data[kFddMaxFrameBytes];
};

struct FddIpcFrameShared
{
    uint32_t magic;
    uint32_t version;
    sem_t mutex;
    sem_t frame_ready;
    uint32_t stop_flag;
    uint64_t latest_seq;
    uint32_t latest_slot;
    FddIpcFrameSlot slots[kFddFrameSlotCount];
};

struct FddIpcResultShared
{
    uint32_t magic;
    uint32_t version;
    sem_t mutex;
    uint32_t stop_flag;
    uint64_t frame_seq;
    uint32_t box_count;
    FddIpcBox boxes[kFddMaxFallBoxes];
};

struct FddIpcContext
{
    int frame_fd = -1;
    int result_fd = -1;
    FddIpcFrameShared *frames = nullptr;
    FddIpcResultShared *result = nullptr;
    bool file_backend = false;
    std::string frame_file;
    std::string result_file;
    std::string stop_file;
};

bool fdd_ipc_open(FddIpcContext &ctx, bool reset_existing, std::string *error);
void fdd_ipc_close(FddIpcContext &ctx);
void fdd_ipc_unlink();

bool fdd_ipc_publish_frame(FddIpcContext &ctx,
                           const uint8_t *data,
                           uint32_t width,
                           uint32_t height,
                           uint32_t channels,
                           uint32_t size,
                           uint64_t timestamp_us,
                           uint64_t *published_seq,
                           std::string *error);

bool fdd_ipc_wait_latest_frame(FddIpcContext &ctx,
                               uint64_t last_seq,
                               int timeout_ms,
                               FddIpcFrameMeta &meta,
                               std::vector<uint8_t> &data);

bool fdd_ipc_publish_result(FddIpcContext &ctx,
                            uint64_t frame_seq,
                            const std::vector<FddIpcBox> &boxes,
                            std::string *error);

bool fdd_ipc_read_result(FddIpcContext &ctx, FddIpcDetectionResult &result);

void fdd_ipc_request_stop(FddIpcContext &ctx);
bool fdd_ipc_should_stop(FddIpcContext &ctx);

#endif
