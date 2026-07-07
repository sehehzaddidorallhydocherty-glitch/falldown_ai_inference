#include "fdd_ipc.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace
{
constexpr const char *kFddFrameFileName = "fdd_fall_frame.bin";
constexpr const char *kFddResultFileName = "fdd_fall_result.bin";
constexpr const char *kFddStopFileName = "fdd_fall_stop.flag";

struct FddFileFrameHeader
{
    uint32_t magic;
    uint32_t version;
    uint64_t seq;
    uint64_t timestamp_us;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t size;
};

struct FddFileResultHeader
{
    uint32_t magic;
    uint32_t version;
    uint64_t frame_seq;
    uint32_t box_count;
    uint32_t stop_flag;
};

std::string errno_message(const char *what)
{
    std::ostringstream oss;
    oss << what << ": " << std::strerror(errno);
    return oss.str();
}

std::string join_path(const std::string &base, const char *name)
{
    if (base.empty() || base == ".")
    {
        return std::string("./") + name;
    }
    if (base.back() == '/')
    {
        return base + name;
    }
    return base + "/" + name;
}

std::string ipc_file_dir()
{
    const char *env_dir = std::getenv("FDD_IPC_DIR");
    if (env_dir && env_dir[0] != '\0')
    {
        return env_dir;
    }
    if (access("/sharefs", W_OK) == 0)
    {
        return "/sharefs";
    }
    return ".";
}

bool write_all(int fd, const void *data, size_t size)
{
    const uint8_t *ptr = static_cast<const uint8_t *>(data);
    size_t written = 0;
    while (written < size)
    {
        const ssize_t ret = write(fd, ptr + written, size - written);
        if (ret <= 0)
        {
            return false;
        }
        written += static_cast<size_t>(ret);
    }
    return true;
}

bool read_all(int fd, void *data, size_t size)
{
    uint8_t *ptr = static_cast<uint8_t *>(data);
    size_t read_size = 0;
    while (read_size < size)
    {
        const ssize_t ret = read(fd, ptr + read_size, size - read_size);
        if (ret <= 0)
        {
            return false;
        }
        read_size += static_cast<size_t>(ret);
    }
    return true;
}

bool write_file_direct(const std::string &path,
                       const void *header,
                       size_t header_size,
                       const void *payload,
                       size_t payload_size,
                       std::string *error)
{
    const int fd = open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0666);
    if (fd < 0)
    {
        if (error)
        {
            *error = errno_message("open IPC file failed");
        }
        return false;
    }

    bool ok = true;
    if (header_size == sizeof(FddFileFrameHeader))
    {
        FddFileFrameHeader pending = *static_cast<const FddFileFrameHeader *>(header);
        pending.size = 0;
        ok = write_all(fd, &pending, sizeof(pending));
    }
    else if (header_size == sizeof(FddFileResultHeader))
    {
        FddFileResultHeader pending = *static_cast<const FddFileResultHeader *>(header);
        pending.box_count = 0;
        pending.stop_flag = 0;
        ok = write_all(fd, &pending, sizeof(pending));
    }
    else
    {
        ok = write_all(fd, header, header_size);
    }

    if (ok && payload && payload_size > 0)
    {
        ok = write_all(fd, payload, payload_size);
    }
    if (ok && lseek(fd, 0, SEEK_SET) < 0)
    {
        ok = false;
    }
    if (ok)
    {
        ok = write_all(fd, header, header_size);
    }

    fsync(fd);
    close(fd);

    if (!ok)
    {
        if (error)
        {
            *error = errno_message("write IPC file failed");
        }
        return false;
    }
    return true;
}

bool write_file_atomic(const std::string &path,
                       const void *header,
                       size_t header_size,
                       const void *payload,
                       size_t payload_size,
                       std::string *error)
{
    return write_file_direct(path, header, header_size, payload, payload_size, error);
}

bool open_file_backend(FddIpcContext &ctx, bool reset_existing, std::string *error)
{
    const std::string dir = ipc_file_dir();
    ctx.file_backend = true;
    ctx.frame_file = join_path(dir, kFddFrameFileName);
    ctx.result_file = join_path(dir, kFddResultFileName);
    ctx.stop_file = join_path(dir, kFddStopFileName);

    if (reset_existing)
    {
        FddFileFrameHeader frame_header{};
        frame_header.magic = kFddFrameMagic;
        frame_header.version = kFddIpcVersion;
        if (!write_file_atomic(ctx.frame_file, &frame_header, sizeof(frame_header), nullptr, 0, error))
        {
            ctx.file_backend = false;
            return false;
        }

        FddFileResultHeader result_header{};
        result_header.magic = kFddResultMagic;
        result_header.version = kFddIpcVersion;
        result_header.stop_flag = 0;
        if (!write_file_atomic(ctx.result_file, &result_header, sizeof(result_header), nullptr, 0, error))
        {
            ctx.file_backend = false;
            return false;
        }
    }
    else if (access(ctx.result_file.c_str(), R_OK) != 0)
    {
        if (error)
        {
            *error = "file IPC result file not found; start falldown_ai_event.elf first";
        }
        ctx.file_backend = false;
        return false;
    }

    std::cout << "[ipc] using file IPC backend under " << dir << std::endl;
    return true;
}

bool map_object(const char *name,
                size_t size,
                bool reset_existing,
                int &fd,
                void **mapped,
                bool &created,
                std::string *error)
{
    created = false;
    if (reset_existing)
    {
        shm_unlink(name);
    }

    int flags = O_RDWR;
    if (reset_existing)
    {
        flags |= O_CREAT | O_EXCL;
    }

    fd = shm_open(name, flags, 0666);
    if (fd < 0 && !reset_existing && errno == ENOENT)
    {
        if (error)
        {
            *error = std::string("shared memory not found: ") + name;
        }
        return false;
    }
    if (fd < 0)
    {
        if (error)
        {
            *error = errno_message("shm_open failed");
        }
        return false;
    }

    created = reset_existing;
    if (created && ftruncate(fd, static_cast<off_t>(size)) != 0)
    {
        if (error)
        {
            *error = errno_message("ftruncate failed");
        }
        close(fd);
        fd = -1;
        return false;
    }

    *mapped = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (*mapped == MAP_FAILED)
    {
        if (error)
        {
            *error = errno_message("mmap failed");
        }
        close(fd);
        fd = -1;
        *mapped = nullptr;
        return false;
    }

    return true;
}

void init_frame_shared(FddIpcFrameShared *shared)
{
    std::memset(shared, 0, sizeof(FddIpcFrameShared));
    sem_init(&shared->mutex, 1, 1);
    sem_init(&shared->frame_ready, 1, 0);
    shared->version = kFddIpcVersion;
    shared->magic = kFddFrameMagic;
}

void init_result_shared(FddIpcResultShared *shared)
{
    std::memset(shared, 0, sizeof(FddIpcResultShared));
    sem_init(&shared->mutex, 1, 1);
    shared->version = kFddIpcVersion;
    shared->magic = kFddResultMagic;
}

bool validate_shared(const FddIpcFrameShared *frames,
                     const FddIpcResultShared *result,
                     std::string *error)
{
    if (!frames || !result ||
        frames->magic != kFddFrameMagic ||
        frames->version != kFddIpcVersion ||
        result->magic != kFddResultMagic ||
        result->version != kFddIpcVersion)
    {
        if (error)
        {
            *error = "shared memory is not initialized or has an incompatible version";
        }
        return false;
    }
    return true;
}
}

bool fdd_ipc_open(FddIpcContext &ctx, bool reset_existing, std::string *error)
{
    fdd_ipc_close(ctx);

    bool frame_created = false;
    bool result_created = false;
    void *frame_map = nullptr;
    void *result_map = nullptr;
    std::string shm_error;

    if (!map_object(kFddFrameShmName,
                    sizeof(FddIpcFrameShared),
                    reset_existing,
                    ctx.frame_fd,
                    &frame_map,
                    frame_created,
                    &shm_error))
    {
        return open_file_backend(ctx, reset_existing, error);
    }

    if (!map_object(kFddResultShmName,
                    sizeof(FddIpcResultShared),
                    reset_existing,
                    ctx.result_fd,
                    &result_map,
                    result_created,
                    &shm_error))
    {
        fdd_ipc_close(ctx);
        return open_file_backend(ctx, reset_existing, error);
    }

    ctx.frames = static_cast<FddIpcFrameShared *>(frame_map);
    ctx.result = static_cast<FddIpcResultShared *>(result_map);

    if (frame_created)
    {
        init_frame_shared(ctx.frames);
    }
    if (result_created)
    {
        init_result_shared(ctx.result);
    }

    if (!validate_shared(ctx.frames, ctx.result, &shm_error))
    {
        fdd_ipc_close(ctx);
        return open_file_backend(ctx, reset_existing, error);
    }

    ctx.file_backend = false;
    return true;
}

void fdd_ipc_close(FddIpcContext &ctx)
{
    if (ctx.frames)
    {
        munmap(ctx.frames, sizeof(FddIpcFrameShared));
        ctx.frames = nullptr;
    }
    if (ctx.result)
    {
        munmap(ctx.result, sizeof(FddIpcResultShared));
        ctx.result = nullptr;
    }
    if (ctx.frame_fd >= 0)
    {
        close(ctx.frame_fd);
        ctx.frame_fd = -1;
    }
    if (ctx.result_fd >= 0)
    {
        close(ctx.result_fd);
        ctx.result_fd = -1;
    }
    ctx.file_backend = false;
    ctx.frame_file.clear();
    ctx.result_file.clear();
    ctx.stop_file.clear();
}

void fdd_ipc_unlink()
{
    shm_unlink(kFddFrameShmName);
    shm_unlink(kFddResultShmName);
}

bool fdd_ipc_publish_frame(FddIpcContext &ctx,
                           const uint8_t *data,
                           uint32_t width,
                           uint32_t height,
                           uint32_t channels,
                           uint32_t size,
                           uint64_t timestamp_us,
                           uint64_t *published_seq,
                           std::string *error)
{
    if (ctx.file_backend)
    {
        if (!data || size == 0 || size > kFddMaxFrameBytes)
        {
            if (error)
            {
                *error = "invalid frame publish request";
            }
            return false;
        }

        uint64_t last_seq = 0;
        FddFileFrameHeader old_header{};
        const int old_fd = open(ctx.frame_file.c_str(), O_RDONLY);
        if (old_fd >= 0)
        {
            if (read_all(old_fd, &old_header, sizeof(old_header)) &&
                old_header.magic == kFddFrameMagic)
            {
                last_seq = old_header.seq;
            }
            close(old_fd);
        }

        FddFileFrameHeader header{};
        header.magic = kFddFrameMagic;
        header.version = kFddIpcVersion;
        header.seq = last_seq + 1;
        header.timestamp_us = timestamp_us;
        header.width = width;
        header.height = height;
        header.channels = channels;
        header.size = size;
        if (!write_file_atomic(ctx.frame_file, &header, sizeof(header), data, size, error))
        {
            return false;
        }
        if (published_seq)
        {
            *published_seq = header.seq;
        }
        return true;
    }

    if (!ctx.frames || !data || size == 0 || size > kFddMaxFrameBytes)
    {
        if (error)
        {
            *error = "invalid frame publish request";
        }
        return false;
    }

    sem_wait(&ctx.frames->mutex);
    const uint64_t seq = ctx.frames->latest_seq + 1;
    const uint32_t slot_index = static_cast<uint32_t>(seq % kFddFrameSlotCount);
    FddIpcFrameSlot &slot = ctx.frames->slots[slot_index];
    slot.seq = seq;
    slot.timestamp_us = timestamp_us;
    slot.width = width;
    slot.height = height;
    slot.channels = channels;
    slot.size = size;
    std::memcpy(slot.data, data, size);
    ctx.frames->latest_slot = slot_index;
    ctx.frames->latest_seq = seq;
    sem_post(&ctx.frames->mutex);
    sem_post(&ctx.frames->frame_ready);

    if (published_seq)
    {
        *published_seq = seq;
    }
    return true;
}

bool fdd_ipc_wait_latest_frame(FddIpcContext &ctx,
                               uint64_t last_seq,
                               int timeout_ms,
                               FddIpcFrameMeta &meta,
                               std::vector<uint8_t> &data)
{
    if (ctx.file_backend)
    {
        const int sleep_us = 10000;
        const int max_attempts = std::max(1, timeout_ms * 1000 / sleep_us);
        for (int attempt = 0; attempt < max_attempts; ++attempt)
        {
            if (fdd_ipc_should_stop(ctx))
            {
                return false;
            }

            const int fd = open(ctx.frame_file.c_str(), O_RDONLY);
            if (fd >= 0)
            {
                FddFileFrameHeader header{};
                bool ok = read_all(fd, &header, sizeof(header));
                if (ok &&
                    header.magic == kFddFrameMagic &&
                    header.version == kFddIpcVersion &&
                    header.seq != last_seq &&
                    header.size > 0 &&
                    header.size <= kFddMaxFrameBytes)
                {
                    data.resize(header.size);
                    ok = read_all(fd, data.data(), header.size);
                    if (ok)
                    {
                        meta.seq = header.seq;
                        meta.timestamp_us = header.timestamp_us;
                        meta.width = header.width;
                        meta.height = header.height;
                        meta.channels = header.channels;
                        meta.size = header.size;
                        close(fd);
                        return true;
                    }
                }
                close(fd);
            }
            usleep(sleep_us);
        }
        return false;
    }

    if (!ctx.frames)
    {
        return false;
    }

    const int sleep_us = 10000;
    const int max_attempts = std::max(1, timeout_ms * 1000 / sleep_us);
    for (int attempt = 0; attempt < max_attempts; ++attempt)
    {
        if (fdd_ipc_should_stop(ctx))
        {
            return false;
        }

        bool copied = false;
        sem_wait(&ctx.frames->mutex);
        if (ctx.frames->latest_seq != last_seq)
        {
            const FddIpcFrameSlot &slot = ctx.frames->slots[ctx.frames->latest_slot];
            if (slot.size > 0 && slot.size <= kFddMaxFrameBytes)
            {
                meta.seq = slot.seq;
                meta.timestamp_us = slot.timestamp_us;
                meta.width = slot.width;
                meta.height = slot.height;
                meta.channels = slot.channels;
                meta.size = slot.size;
                data.assign(slot.data, slot.data + slot.size);
                copied = true;
            }
        }
        sem_post(&ctx.frames->mutex);

        if (copied)
        {
            return true;
        }
        usleep(sleep_us);
    }
    return false;
}

bool fdd_ipc_publish_result(FddIpcContext &ctx,
                            uint64_t frame_seq,
                            const std::vector<FddIpcBox> &boxes,
                            std::string *error)
{
    if (ctx.file_backend)
    {
        uint32_t existing_stop_flag = 0;
        const int old_fd = open(ctx.result_file.c_str(), O_RDONLY);
        if (old_fd >= 0)
        {
            FddFileResultHeader old_header{};
            if (read_all(old_fd, &old_header, sizeof(old_header)) &&
                old_header.magic == kFddResultMagic &&
                old_header.version == kFddIpcVersion)
            {
                existing_stop_flag = old_header.stop_flag;
            }
            close(old_fd);
        }

        FddFileResultHeader header{};
        header.magic = kFddResultMagic;
        header.version = kFddIpcVersion;
        header.frame_seq = frame_seq;
        header.box_count = static_cast<uint32_t>(std::min<size_t>(boxes.size(), kFddMaxFallBoxes));
        header.stop_flag = existing_stop_flag;
        return write_file_atomic(ctx.result_file,
                                 &header,
                                 sizeof(header),
                                 boxes.data(),
                                 sizeof(FddIpcBox) * header.box_count,
                                 error);
    }

    if (!ctx.result)
    {
        if (error)
        {
            *error = "result shared memory is not open";
        }
        return false;
    }

    sem_wait(&ctx.result->mutex);
    ctx.result->frame_seq = frame_seq;
    ctx.result->box_count = static_cast<uint32_t>(std::min<size_t>(boxes.size(), kFddMaxFallBoxes));
    for (uint32_t i = 0; i < ctx.result->box_count; ++i)
    {
        ctx.result->boxes[i] = boxes[i];
    }
    sem_post(&ctx.result->mutex);
    return true;
}

bool fdd_ipc_read_result(FddIpcContext &ctx, FddIpcDetectionResult &result)
{
    if (ctx.file_backend)
    {
        const int fd = open(ctx.result_file.c_str(), O_RDONLY);
        if (fd < 0)
        {
            return false;
        }

        FddFileResultHeader header{};
        bool ok = read_all(fd, &header, sizeof(header));
        if (!ok ||
            header.magic != kFddResultMagic ||
            header.version != kFddIpcVersion ||
            header.box_count > kFddMaxFallBoxes)
        {
            close(fd);
            return false;
        }

        result.frame_seq = header.frame_seq;
        result.boxes.resize(header.box_count);
        if (header.box_count > 0)
        {
            ok = read_all(fd, result.boxes.data(), sizeof(FddIpcBox) * header.box_count);
        }
        close(fd);
        return ok;
    }

    if (!ctx.result)
    {
        return false;
    }

    sem_wait(&ctx.result->mutex);
    result.frame_seq = ctx.result->frame_seq;
    const uint32_t count = std::min<uint32_t>(ctx.result->box_count, kFddMaxFallBoxes);
    result.boxes.assign(ctx.result->boxes, ctx.result->boxes + count);
    sem_post(&ctx.result->mutex);
    return true;
}

void fdd_ipc_request_stop(FddIpcContext &ctx)
{
    if (ctx.file_backend)
    {
        const int fd = open(ctx.result_file.c_str(), O_RDWR);
        if (fd >= 0)
        {
            FddFileResultHeader header{};
            if (read_all(fd, &header, sizeof(header)) &&
                header.magic == kFddResultMagic &&
                header.version == kFddIpcVersion)
            {
                header.stop_flag = 1;
                lseek(fd, 0, SEEK_SET);
                write_all(fd, &header, sizeof(header));
                fsync(fd);
            }
            close(fd);
        }
        return;
    }

    if (ctx.frames)
    {
        sem_wait(&ctx.frames->mutex);
        ctx.frames->stop_flag = 1;
        sem_post(&ctx.frames->mutex);
        sem_post(&ctx.frames->frame_ready);
    }
    if (ctx.result)
    {
        sem_wait(&ctx.result->mutex);
        ctx.result->stop_flag = 1;
        sem_post(&ctx.result->mutex);
    }
}

bool fdd_ipc_should_stop(FddIpcContext &ctx)
{
    if (ctx.file_backend)
    {
        const int fd = open(ctx.result_file.c_str(), O_RDONLY);
        if (fd < 0)
        {
            return false;
        }
        FddFileResultHeader header{};
        const bool ok = read_all(fd, &header, sizeof(header));
        close(fd);
        return ok &&
               header.magic == kFddResultMagic &&
               header.version == kFddIpcVersion &&
               header.stop_flag != 0;
    }

    bool stop = false;
    if (ctx.frames)
    {
        sem_wait(&ctx.frames->mutex);
        stop = stop || ctx.frames->stop_flag != 0;
        sem_post(&ctx.frames->mutex);
    }
    if (ctx.result)
    {
        sem_wait(&ctx.result->mutex);
        stop = stop || ctx.result->stop_flag != 0;
        sem_post(&ctx.result->mutex);
    }
    return stop;
}
