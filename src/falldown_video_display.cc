#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <memory>
#include <string>
#include <sys/select.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "ai_inference.h"
#include "comm_server.h"
#include "fdd_ipc.h"
#include "vi_vo.h"

namespace
{
volatile sig_atomic_t g_stop = 0;

constexpr int kPreEventSeconds = 10;
constexpr int kFallVoteWindowSeconds = 6;
constexpr int kFallVoteMinHits = 3;
constexpr int kEventResetSeconds = 3;
constexpr int kEvidenceJpegQuality = 70;
constexpr int kEvidenceMaxWidth = 640;
constexpr int kFrameSaveProgressInterval = 10;
constexpr double kDefaultIspEvidenceFps = 8.0;
const char *kDefaultEvidenceRoot = "/sharefs/sdcard/evidence";
const char *kFallbackEvidenceRoot = "/sdcard/evidence";
const char *kLocalEvidenceRoot = "evidence";

struct FallBox
{
    bool valid = false;
    double x1 = 0.0;
    double y1 = 0.0;
    double x2 = 0.0;
    double y2 = 0.0;
};

struct EvidenceFrame
{
    std::chrono::steady_clock::time_point captured_at;
    std::vector<uchar> encoded;
};

struct FallAlarmState
{
    std::mutex mutex;
    std::deque<std::chrono::steady_clock::time_point> fall_hit_times;
    std::deque<EvidenceFrame> frame_buffer;
    std::chrono::steady_clock::time_point clear_since;
    std::chrono::steady_clock::time_point last_debug_print;
    bool event_latched = false;
    bool clear_started = false;
};

class FallCommCallback : public IServerCallback
{
public:
    void OnMessage(const UserMessage *message) override
    {
        if (!message)
        {
            return;
        }
        std::cout << "[fall-comm] client message type=" << message->type << std::endl;
    }

    void OnAEncFrameData(const AVEncFrameData &) override
    {
    }
};

class FallCommNotifier
{
public:
    ~FallCommNotifier()
    {
        Stop();
    }

    void Start()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (started_)
        {
            return;
        }
        started_ = true;
        init_thread_ = std::thread([this]() {
            std::unique_ptr<UserCommServer> server(new UserCommServer());
            if (server->Init("falldown", &callback_) < 0)
            {
                std::cerr << "[fall-comm] UserCommServer init failed; Linux notify disabled" << std::endl;
                std::lock_guard<std::mutex> lock(mutex_);
                started_ = false;
                return;
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                server_ = std::move(server);
                ready_ = true;
            }
            std::cout << "[fall-comm] Linux notify client connected" << std::endl;
        });
        init_thread_.detach();
    }

    void Stop()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ready_ = false;
            started_ = false;
        }

        std::unique_ptr<UserCommServer> server;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            server = std::move(server_);
        }

        if (server)
        {
            server->DeInit();
        }
    }

    void NotifyFallEvent(const std::string &event_dir, const std::string &snapshot_path, int frame_count)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ready_ || !server_)
        {
            return;
        }

        char json[512] = {0};
        std::snprintf(json,
                      sizeof(json),
                      "{\"type\":\"fall_detected\",\"event_dir\":\"%s\",\"snapshot\":\"%s\",\"frame_count\":%d}",
                      event_dir.c_str(),
                      snapshot_path.c_str(),
                      frame_count);

        const size_t json_len = std::strlen(json);
        std::vector<uint8_t> buffer(sizeof(UserMessage) + json_len + 1);
        UserMessage *message = reinterpret_cast<UserMessage *>(buffer.data());
        message->type = static_cast<int>(UserMsgType::SERVER_MSG_BASE);
        message->len = static_cast<int>(json_len);
        std::memcpy(message->data, json, json_len + 1);
        if (server_->SendMessage(buffer.data(), static_cast<int>(buffer.size())) < 0)
        {
            std::cerr << "[fall-comm] failed to send fall event to Linux client" << std::endl;
        }
    }

private:
    std::mutex mutex_;
    FallCommCallback callback_;
    std::unique_ptr<UserCommServer> server_;
    std::thread init_thread_;
    bool started_ = false;
    bool ready_ = false;
};

FallCommNotifier g_fall_comm_notifier;

void handle_signal(int)
{
    g_stop = 1;
}

void print_usage(const char *name)
{
    std::cout << "Usage: " << name << " <debug_mode> [display_mode]" << std::endl
              << "For example:" << std::endl
              << "  ./" << name << " 0" << std::endl
              << "  ./" << name << " 0 off" << std::endl
              << "display_mode: auto | on | off" << std::endl;
}

bool valid_display_mode(const char *value)
{
    return value != nullptr &&
           (std::strcmp(value, "auto") == 0 ||
            std::strcmp(value, "on") == 0 ||
            std::strcmp(value, "off") == 0);
}

double env_double_or_default(const char *name, double default_value, double min_value, double max_value)
{
    const char *value = std::getenv(name);
    if (!value || value[0] == '\0')
    {
        return default_value;
    }

    char *end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (end == value || *end != '\0' || parsed < min_value || parsed > max_value)
    {
        std::cerr << "warning: ignore invalid " << name << "=" << value
                  << ", expected " << min_value << ".." << max_value << std::endl;
        return default_value;
    }
    return parsed;
}

bool keyboard_quit_requested()
{
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(STDIN_FILENO, &readfds);
    timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 0;

    const int ret = select(STDIN_FILENO + 1, &readfds, nullptr, nullptr, &timeout);
    if (ret > 0 && FD_ISSET(STDIN_FILENO, &readfds))
    {
        char ch = 0;
        if (read(STDIN_FILENO, &ch, 1) == 1 && ch == 'q')
        {
            return true;
        }
    }
    return false;
}

std::string join_path(const std::string &base, const std::string &child)
{
    if (base.empty())
    {
        return child;
    }
    if (base.back() == '/' || base.back() == '\\')
    {
        return base + child;
    }
    return base + "/" + child;
}

int ensure_directory(const std::string &path)
{
    if (path.empty())
    {
        return -1;
    }

    struct stat path_stat;
    if (stat(path.c_str(), &path_stat) == 0)
    {
        return S_ISDIR(path_stat.st_mode) ? 0 : -1;
    }

    if (mkdir(path.c_str(), 0755) == 0 || errno == EEXIST)
    {
        return 0;
    }
    std::cerr << "failed to create directory: " << path << std::endl;
    return -1;
}

std::string timestamp_string()
{
    std::time_t now = std::time(nullptr);
    std::tm tm_value;
    localtime_r(&now, &tm_value);

    char buffer[32] = {0};
    std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &tm_value);
    return buffer;
}

std::string prepare_evidence_root()
{
    const char *env_root = std::getenv("FDD_EVIDENCE_ROOT");
    const char *candidates[] = {
        env_root && env_root[0] != '\0' ? env_root : nullptr,
        kDefaultEvidenceRoot,
        kFallbackEvidenceRoot,
        kLocalEvidenceRoot,
    };

    for (const char *candidate : candidates)
    {
        if (!candidate)
        {
            continue;
        }
        std::string root(candidate);
        if (ensure_directory(root) == 0)
        {
            return root;
        }
    }
    return "";
}

std::string create_unique_event_dir(const std::string &root)
{
    const std::string base = join_path(root, timestamp_string());
    for (int attempt = 0; attempt < 1000; ++attempt)
    {
        std::string event_dir = base;
        if (attempt > 0)
        {
            event_dir += "_";
            event_dir += std::to_string(attempt);
        }
        if (mkdir(event_dir.c_str(), 0755) == 0)
        {
            return event_dir;
        }
        if (errno != EEXIST)
        {
            break;
        }
    }
    return "";
}

void draw_fall_label(cv::Mat &image, int x1, int y1)
{
    if (image.empty())
    {
        return;
    }

    constexpr const char *kFallOsdText = "Fall Detected!";
    constexpr int kFontFace = cv::FONT_HERSHEY_SIMPLEX;
    constexpr double kFontScale = 0.8;
    constexpr int kTextThickness = 2;
    constexpr int kPaddingX = 8;
    constexpr int kPaddingY = 6;
    constexpr int kLabelMargin = 4;

    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(kFallOsdText, kFontFace, kFontScale, kTextThickness, &baseline);
    const int raw_label_w = text_size.width + kPaddingX * 2;
    const int raw_label_h = text_size.height + baseline + kPaddingY * 2;
    if (raw_label_w <= 0 || raw_label_h <= 0 || image.cols <= 0 || image.rows <= 0)
    {
        return;
    }

    const int label_w = std::min(raw_label_w, image.cols);
    const int label_h = std::min(raw_label_h, image.rows);
    int label_x = std::max(0, std::min(x1, image.cols - label_w));
    int label_y = y1 - label_h - kLabelMargin;
    if (label_y < 0)
    {
        label_y = y1 + kLabelMargin;
    }
    label_y = std::max(0, std::min(label_y, image.rows - label_h));

    const bool has_alpha = image.channels() == 4;
    const cv::Scalar background = has_alpha ? cv::Scalar(0, 0, 180, 220) : cv::Scalar(0, 0, 180);
    const cv::Scalar text_color = has_alpha ? cv::Scalar(255, 255, 255, 255) : cv::Scalar(255, 255, 255);

    cv::rectangle(image, cv::Rect(label_x, label_y, label_w, label_h), background, -1, cv::LINE_8, 0);
    cv::putText(image,
                kFallOsdText,
                cv::Point(label_x + kPaddingX, label_y + kPaddingY + text_size.height),
                kFontFace,
                kFontScale,
                text_color,
                kTextThickness,
                cv::LINE_AA);
}

void draw_boxes_scaled(cv::Mat &image,
                       const DetectResult &result,
                       int src_width,
                       int src_height,
                       int debug_mode,
                       const cv::Scalar &color)
{
    for (const auto &r : result.boxes)
    {
        ScopedTiming st("draw boxes", debug_mode);
        const int x1 = r.x1 * image.cols / src_width;
        const int y1 = r.y1 * image.rows / src_height;
        const int w = (r.x2 - r.x1) * image.cols / src_width;
        const int h = (r.y2 - r.y1) * image.rows / src_height;
        cv::rectangle(image, cv::Rect(x1, y1, w, h), color, 6, 2, 0);
        draw_fall_label(image, x1, y1);
    }
}

cv::Mat rgb_chw_to_bgr_mat(const uint8_t *data, int width, int height)
{
    cv::Mat image(height, width, CV_8UC3);
    const size_t plane_size = static_cast<size_t>(width) * height;
    const uint8_t *r_plane = data;
    const uint8_t *g_plane = data + plane_size;
    const uint8_t *b_plane = data + plane_size * 2;

    for (int y = 0; y < height; ++y)
    {
        cv::Vec3b *row = image.ptr<cv::Vec3b>(y);
        const size_t row_offset = static_cast<size_t>(y) * width;
        for (int x = 0; x < width; ++x)
        {
            const size_t idx = row_offset + x;
            row[x] = cv::Vec3b(b_plane[idx], g_plane[idx], r_plane[idx]);
        }
    }
    return image;
}

DetectResult to_detect_result(const FddIpcDetectionResult &ipc_result)
{
    DetectResult result;
    result.boxes.clear();
    result.landmarks.clear();
    for (const auto &ipc_box : ipc_result.boxes)
    {
        face_coordinate box;
        box.x1 = ipc_box.x1;
        box.y1 = ipc_box.y1;
        box.x2 = ipc_box.x2;
        box.y2 = ipc_box.y2;
        result.boxes.push_back(box);
    }
    return result;
}

FallBox largest_fall_box(const DetectResult &result)
{
    FallBox best;
    double best_area = 0.0;
    for (const auto &box : result.boxes)
    {
        const double width = std::max(0, box.x2 - box.x1);
        const double height = std::max(0, box.y2 - box.y1);
        const double area = width * height;
        if (area > best_area)
        {
            best_area = area;
            best.valid = true;
            best.x1 = box.x1;
            best.y1 = box.y1;
            best.x2 = box.x2;
            best.y2 = box.y2;
        }
    }
    return best;
}

bool encode_evidence_frame(const cv::Mat &frame, std::vector<uchar> &encoded_frame)
{
    encoded_frame.clear();
    if (frame.empty())
    {
        return false;
    }

    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, kEvidenceJpegQuality};
    cv::Mat frame_to_encode = frame;
    cv::Mat resized_frame;
    if (frame.cols > kEvidenceMaxWidth)
    {
        const double scale = static_cast<double>(kEvidenceMaxWidth) / frame.cols;
        const int resized_height = std::max(1, static_cast<int>(frame.rows * scale + 0.5));
        cv::resize(frame, resized_frame, cv::Size(kEvidenceMaxWidth, resized_height));
        frame_to_encode = resized_frame;
    }
    return cv::imencode(".jpg", frame_to_encode, encoded_frame, params);
}

void cache_evidence_frame(FallAlarmState &alarm,
                          const cv::Mat &frame,
                          std::chrono::steady_clock::time_point now)
{
    std::vector<uchar> encoded_frame;
    if (!encode_evidence_frame(frame, encoded_frame))
    {
        return;
    }

    std::lock_guard<std::mutex> lock(alarm.mutex);
    alarm.frame_buffer.push_back({now, encoded_frame});
    const auto oldest_to_keep = now - std::chrono::seconds(kPreEventSeconds);
    while (!alarm.frame_buffer.empty() && alarm.frame_buffer.front().captured_at < oldest_to_keep)
    {
        alarm.frame_buffer.pop_front();
    }
}

std::vector<EvidenceFrame> ordered_cached_frames(FallAlarmState &alarm)
{
    std::lock_guard<std::mutex> lock(alarm.mutex);
    std::vector<EvidenceFrame> frames;
    frames.reserve(alarm.frame_buffer.size());
    for (const auto &frame : alarm.frame_buffer)
    {
        if (!frame.encoded.empty())
        {
            frames.push_back(frame);
        }
    }
    return frames;
}

bool save_frame_sequence(const std::vector<EvidenceFrame> &frames, const std::string &event_dir)
{
    const std::string frames_dir = join_path(event_dir, "frames");
    if (ensure_directory(frames_dir) != 0)
    {
        return false;
    }

    int saved_count = 0;
    for (size_t i = 0; i < frames.size(); ++i)
    {
        char file_name[32] = {0};
        std::snprintf(file_name, sizeof(file_name), "%06d.jpg", static_cast<int>(i + 1));
        const std::string frame_path = join_path(frames_dir, file_name);
        std::ofstream frame_file(frame_path, std::ios::binary);
        if (!frame_file)
        {
            continue;
        }
        frame_file.write(reinterpret_cast<const char *>(frames[i].encoded.data()),
                         static_cast<std::streamsize>(frames[i].encoded.size()));
        if (frame_file.good())
        {
            ++saved_count;
            if (saved_count % kFrameSaveProgressInterval == 0)
            {
                std::cout << "fall evidence frame sequence progress: "
                          << saved_count << "/" << frames.size() << std::endl;
            }
        }
    }

    std::cout << "fall evidence frame sequence saved: " << frames_dir
              << " frames=" << saved_count << std::endl;
    return saved_count > 0;
}

void save_fall_event_worker(const std::vector<EvidenceFrame> &frames)
{
    const std::string root = prepare_evidence_root();
    if (root.empty())
    {
        std::cerr << "warning: fall evidence root unavailable" << std::endl;
        return;
    }

    const std::string event_dir = create_unique_event_dir(root);
    if (event_dir.empty())
    {
        std::cerr << "warning: failed to create fall event directory" << std::endl;
        return;
    }

    const std::string snapshot_path = join_path(event_dir, "snapshot.jpg");
    if (!frames.empty() && !frames.back().encoded.empty())
    {
        std::ofstream snapshot_file(snapshot_path, std::ios::binary);
        snapshot_file.write(reinterpret_cast<const char *>(frames.back().encoded.data()),
                            static_cast<std::streamsize>(frames.back().encoded.size()));
        if (snapshot_file.good())
        {
            std::cout << "fall alarm snapshot saved: " << snapshot_path << std::endl;
        }
    }

    save_frame_sequence(frames, event_dir);
    g_fall_comm_notifier.NotifyFallEvent(event_dir, snapshot_path, static_cast<int>(frames.size()));
}

void start_fall_event_save(FallAlarmState &alarm)
{
    std::vector<EvidenceFrame> frames = ordered_cached_frames(alarm);
    if (frames.empty())
    {
        std::cerr << "warning: skip fall evidence save because frame cache is empty" << std::endl;
        return;
    }

    std::cout << "fall evidence save queued: frames=" << frames.size() << std::endl;
    std::thread saver([frames = std::move(frames)]() {
        save_fall_event_worker(frames);
    });
    saver.detach();
}

void update_fall_alarm(FallAlarmState &alarm, const cv::Mat &evidence_frame, const DetectResult &result)
{
    const auto now = std::chrono::steady_clock::now();
    if (!evidence_frame.empty())
    {
        cache_evidence_frame(alarm, evidence_frame, now);
    }

    const FallBox current_box = largest_fall_box(result);
    const auto vote_window_start = now - std::chrono::seconds(kFallVoteWindowSeconds);
    bool trigger_save = false;

    {
        std::lock_guard<std::mutex> lock(alarm.mutex);
        while (!alarm.fall_hit_times.empty() && alarm.fall_hit_times.front() < vote_window_start)
        {
            alarm.fall_hit_times.pop_front();
        }

        if (!current_box.valid)
        {
            if (!alarm.clear_started)
            {
                alarm.clear_started = true;
                alarm.clear_since = now;
            }
            if (alarm.event_latched && now - alarm.clear_since >= std::chrono::seconds(kEventResetSeconds))
            {
                alarm.event_latched = false;
                alarm.fall_hit_times.clear();
            }
            return;
        }

        alarm.clear_started = false;
        alarm.fall_hit_times.push_back(now);
        if (!alarm.event_latched && alarm.fall_hit_times.size() >= kFallVoteMinHits)
        {
            alarm.event_latched = true;
            trigger_save = true;
        }
    }

    if (trigger_save)
    {
        start_fall_event_save(alarm);
    }
}

bool local_display_needs_rotate()
{
#if defined(CONFIG_BOARD_K230D_CANMV) || defined(CONFIG_BOARD_K230_CANMV_DONGSHANPI) || defined(CONFIG_BOARD_K230_CANMV_01STUDIO)
    return true;
#else
    return false;
#endif
}

uint64_t now_microseconds()
{
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}

bool read_isp_frame(std::vector<uint8_t> &frame_chw, size_t frame_size, int debug_mode)
{
    k_video_frame_info local_dump_info;
    std::memset(&local_dump_info, 0, sizeof(k_video_frame_info));

    {
        ScopedTiming st("read capture", debug_mode);
        int ret = kd_mpi_vicap_dump_frame(vicap_dev, VICAP_CHN_ID_1, VICAP_DUMP_YUV, &local_dump_info, 1000);
        if (ret)
        {
            printf("sample_vicap...kd_mpi_vicap_dump_frame failed.\n");
            return false;
        }
    }

    bool copied = false;
    {
        ScopedTiming st("isp copy", debug_mode);
        auto vbvaddr = kd_mpi_sys_mmap_cached(local_dump_info.v_frame.phys_addr[0], frame_size);
        if (vbvaddr != nullptr)
        {
            std::memcpy(frame_chw.data(), (void *)vbvaddr, frame_size);
            kd_mpi_sys_munmap(vbvaddr, frame_size);
            copied = true;
        }
        else
        {
            printf("sample_vicap...kd_mpi_sys_mmap_cached failed.\n");
        }
    }

    int ret = kd_mpi_vicap_dump_release(vicap_dev, VICAP_CHN_ID_1, &local_dump_info);
    if (ret)
    {
        printf("sample_vicap...kd_mpi_vicap_dump_release failed.\n");
    }
    return copied;
}
}

int main(int argc, char *argv[])
{
    std::cout << "falldown_video_display built at " << __DATE__ << " " << __TIME__ << std::endl;
    if (argc != 2 && argc != 3)
    {
        print_usage(argv[0]);
        return -1;
    }

    if (argc == 3)
    {
        if (!valid_display_mode(argv[2]))
        {
            std::cerr << "[video-display] invalid display_mode: " << argv[2] << std::endl;
            print_usage(argv[0]);
            return -1;
        }
        setenv("FDD_DISPLAY", argv[2], 1);
        std::cout << "[video-display] FDD_DISPLAY=" << argv[2] << std::endl;
    }

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    const int debug_mode = std::atoi(argv[1]);
    const double evidence_fps = env_double_or_default("FDD_ISP_EVIDENCE_FPS",
                                                     kDefaultIspEvidenceFps,
                                                     1.0,
                                                     15.0);
    const auto evidence_interval = std::chrono::duration<double>(1.0 / evidence_fps);

    FddIpcContext ipc;
    std::string error;
    bool ipc_ready = false;
    for (int attempt = 0; attempt < 50; ++attempt)
    {
        error.clear();
        if (fdd_ipc_open(ipc, false, &error))
        {
            ipc_ready = true;
            break;
        }
        usleep(100000);
    }
    if (!ipc_ready)
    {
        std::cerr << "[video-display] failed to open IPC: " << error << std::endl
                  << "[video-display] start falldown_ai_event.elf first" << std::endl;
        return -1;
    }

    if (vivcap_start() != 0)
    {
        std::cerr << "[video-display] failed to start VICAP/display pipeline" << std::endl;
        fdd_ipc_request_stop(ipc);
        fdd_ipc_close(ipc);
        return -1;
    }

    k_video_frame_info vf_info;
    void *pic_vaddr = NULL;
    bool display_ready = FDD_DISPLAY_IS_ENABLED();
    std::memset(&vf_info, 0, sizeof(vf_info));
    if (display_ready)
    {
        vf_info.v_frame.width = osd_width;
        vf_info.v_frame.height = osd_height;
        vf_info.v_frame.stride[0] = osd_width;
        vf_info.v_frame.pixel_format = PIXEL_FORMAT_ARGB_8888;
        block = vo_insert_frame(&vf_info, &pic_vaddr);
        if (block == VB_INVALID_HANDLE || pic_vaddr == NULL)
        {
            std::cerr << "[video-display] failed to allocate OSD frame; continue headless" << std::endl;
            display_ready = false;
        }
    }
    std::cout << "[video-display] display: "
              << (display_ready ? "enabled" : "headless") << std::endl;
    g_fall_comm_notifier.Start();

    const bool rotate_osd_for_draw = local_display_needs_rotate();
    const size_t frame_size = SENSOR_CHANNEL * SENSOR_HEIGHT * SENSOR_WIDTH;
    std::vector<uint8_t> frame_chw(frame_size);
    FallAlarmState alarm;
    DetectResult latest_result;
    uint64_t latest_result_seq = 0;
    uint64_t last_alarm_result_seq = 0;
    auto last_evidence_cache = std::chrono::steady_clock::time_point{};

    std::cout << "[video-display] running; press q to exit" << std::endl;
    while (!g_stop && !fdd_ipc_should_stop(ipc))
    {
        if (keyboard_quit_requested())
        {
            break;
        }

        const auto loop_start = std::chrono::steady_clock::now();
        if (!read_isp_frame(frame_chw, frame_size, debug_mode))
        {
            continue;
        }

        if (!fdd_ipc_publish_frame(ipc,
                                   frame_chw.data(),
                                   SENSOR_WIDTH,
                                   SENSOR_HEIGHT,
                                   SENSOR_CHANNEL,
                                   static_cast<uint32_t>(frame_size),
                                   now_microseconds(),
                                   nullptr,
                                   &error))
        {
            std::cerr << "[video-display] failed to publish frame: " << error << std::endl;
            continue;
        }

        FddIpcDetectionResult ipc_result;
        if (fdd_ipc_read_result(ipc, ipc_result) && ipc_result.frame_seq != 0)
        {
            latest_result = to_detect_result(ipc_result);
            latest_result_seq = ipc_result.frame_seq;
        }

        const auto now = std::chrono::steady_clock::now();
        if (last_evidence_cache.time_since_epoch().count() == 0 ||
            now - last_evidence_cache >= evidence_interval)
        {
            cv::Mat evidence_frame = rgb_chw_to_bgr_mat(frame_chw.data(), SENSOR_WIDTH, SENSOR_HEIGHT);
            draw_boxes_scaled(evidence_frame,
                              latest_result,
                              SENSOR_WIDTH,
                              SENSOR_HEIGHT,
                              debug_mode,
                              cv::Scalar(0, 0, 255));
            cache_evidence_frame(alarm, evidence_frame, now);
            last_evidence_cache = now;
        }

        if (latest_result_seq != 0 && latest_result_seq != last_alarm_result_seq)
        {
            cv::Mat trigger_evidence_frame;
            if (!latest_result.boxes.empty())
            {
                trigger_evidence_frame = rgb_chw_to_bgr_mat(frame_chw.data(), SENSOR_WIDTH, SENSOR_HEIGHT);
                draw_boxes_scaled(trigger_evidence_frame,
                                  latest_result,
                                  SENSOR_WIDTH,
                                  SENSOR_HEIGHT,
                                  debug_mode,
                                  cv::Scalar(0, 0, 255));
            }
            update_fall_alarm(alarm, trigger_evidence_frame, latest_result);
            last_alarm_result_seq = latest_result_seq;
        }

        if (display_ready)
        {
            cv::Mat osd_frame(osd_height, osd_width, CV_8UC4, cv::Scalar(0, 0, 0, 0));
            cv::Mat draw_frame = osd_frame;
            if (rotate_osd_for_draw)
            {
                cv::rotate(osd_frame, draw_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
            }

            draw_boxes_scaled(draw_frame,
                              latest_result,
                              SENSOR_WIDTH,
                              SENSOR_HEIGHT,
                              debug_mode,
                              cv::Scalar(255, 0, 0, 255));

            if (rotate_osd_for_draw)
            {
                cv::rotate(draw_frame, osd_frame, cv::ROTATE_90_CLOCKWISE);
            }

            {
                ScopedTiming st("osd copy", debug_mode);
                std::memcpy(pic_vaddr, osd_frame.data, osd_width * osd_height * 4);
                kd_mpi_vo_chn_insert_frame(osd_id + 3, &vf_info);
            }
        }

        const auto loop_stop = std::chrono::steady_clock::now();
        const auto elapsed = loop_stop - loop_start;
        if (elapsed < evidence_interval)
        {
            const auto sleep_us = std::chrono::duration_cast<std::chrono::microseconds>(evidence_interval - elapsed).count();
            if (sleep_us > 0)
            {
                usleep(static_cast<unsigned int>(sleep_us));
            }
        }
    }

    fdd_ipc_request_stop(ipc);
    fdd_ipc_close(ipc);
    g_fall_comm_notifier.Stop();
    if (display_ready)
    {
        vo_osd_release_block();
    }
    vivcap_stop();
    std::cout << "[video-display] stopped" << std::endl;
    return 0;
}
