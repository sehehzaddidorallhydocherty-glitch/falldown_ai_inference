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
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <cmath>
#include <memory>
#include <mutex>
#include <sstream>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <algorithm>
#include <cctype>
#include <deque>
#include <thread>
#include <string>
#include <utility>
#include <vector>
#include <opencv2/videoio.hpp>
#include "vi_vo.h"
#include "ai_inference.h"

using std::cerr;
using std::cout;
using std::endl;

std::atomic<bool> isp_stop(false);

void print_usage(const char *name)
{
    cout << "Usage: " << name << " <kmodel> <input_mode> <debug_mode>" << endl
         << "For example: " << endl
         << " [for img] ./falldown_detect.elf yolov5n-falldown.kmodel falldown_elder.jpg 0" << endl
         << " [for video] ./falldown_detect.elf yolov5n-falldown.kmodel falldown.avi 0" << endl
         << " [for frames] ./falldown_detect.elf yolov5n-falldown.kmodel frames_dir 0" << endl
         << " [for isp] ./falldown_detect.elf yolov5n-falldown.kmodel None 0" << endl
         << "Options:" << endl
         << " 1> kmodel      falldown kmodel path\n"
         << " 2> input_mode  local image path / video path / frame directory / camera(None)\n"
         << " 3> debug_mode  0:disable profile, 1:simple timing, 2:verbose timing\n"
         << "\n"
         << endl;
}
static bool has_video_extension(const std::string &path)
{
    const size_t dot_pos = path.find_last_of('.');
    if (dot_pos == std::string::npos)
    {
        return false;
    }

    std::string ext = path.substr(dot_pos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });

    return ext == "mp4" || ext == "avi" || ext == "mov" || ext == "mkv" ||
           ext == "m4v" || ext == "flv" || ext == "wmv";
}

static bool has_image_extension(const std::string &path)
{
    const size_t dot_pos = path.find_last_of('.');
    if (dot_pos == std::string::npos)
    {
        return false;
    }

    std::string ext = path.substr(dot_pos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });

    return ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp";
}

static bool is_directory_path(const std::string &path)
{
    struct stat path_stat;
    if (stat(path.c_str(), &path_stat) != 0)
    {
        return false;
    }
    return S_ISDIR(path_stat.st_mode);
}

static std::vector<std::string> list_frame_images(const std::string &dir_path)
{
    std::vector<std::string> frames;
    DIR *dir = opendir(dir_path.c_str());
    if (!dir)
    {
        std::cerr << "failed to open frame directory: " << dir_path << std::endl;
        return frames;
    }

    struct dirent *entry = nullptr;
    while ((entry = readdir(dir)) != nullptr)
    {
        std::string name = entry->d_name;
        if (name == "." || name == "..")
        {
            continue;
        }
        if (!has_image_extension(name))
        {
            continue;
        }
        frames.push_back(dir_path + "/" + name);
    }
    closedir(dir);
    std::sort(frames.begin(), frames.end());
    return frames;
}

namespace
{
constexpr int kPreEventSeconds = 10;
constexpr int kFallVoteWindowSeconds = 6;
constexpr int kFallVoteMinHits = 3;
constexpr int kEventResetSeconds = 3;
constexpr int kEvidenceJpegQuality = 70;
constexpr int kEvidenceMaxWidth = 640;
constexpr int kFrameSaveProgressInterval = 10;
constexpr double kDefaultEvidenceFps = 25.0;
constexpr double kDefaultIspEvidenceFps = 8.0;
const char *kDefaultEvidenceRoot = "/sharefs/sdcard/evidence";
const char *kFallbackEvidenceRoot = "/sdcard/evidence";
const char *kLocalEvidenceRoot = "evidence";

int ensure_directory(const std::string &path)
{
    if (path.empty())
    {
        return -1;
    }

    struct stat path_stat;
    if (stat(path.c_str(), &path_stat) == 0)
    {
        if (S_ISDIR(path_stat.st_mode))
        {
            return 0;
        }
        std::cerr << "path exists but is not a directory: " << path << std::endl;
        return -1;
    }

    if (mkdir(path.c_str(), 0755) == 0 || errno == EEXIST)
    {
        return 0;
    }
    std::cerr << "failed to create directory: " << path << std::endl;
    return -1;
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

std::string prepare_evidence_root()
{
    const char *env_root = std::getenv("FDD_EVIDENCE_ROOT");
    if (env_root && env_root[0] != '\0')
    {
        const std::string root(env_root);
        if (ensure_directory(root) == 0)
        {
            return root;
        }
        std::cerr << "warning: FDD_EVIDENCE_ROOT unavailable: " << root << std::endl;
        return "";
    }

    const char *candidates[] = {
        kDefaultEvidenceRoot,
        kFallbackEvidenceRoot,
        kLocalEvidenceRoot,
    };

    for (const char *candidate : candidates)
    {
        const std::string root(candidate);
        if (ensure_directory(root) == 0)
        {
            if (root != kDefaultEvidenceRoot)
            {
                std::cout << "fall evidence root selected: " << root << std::endl;
            }
            return root;
        }
    }

    std::cerr << "warning: no writable fall evidence root found" << std::endl;
    return "";
}

std::string timestamp_string()
{
    std::time_t now = std::time(nullptr);
    std::tm local_time;
#if defined(_WIN32)
    localtime_s(&local_time, &now);
#else
    localtime_r(&now, &local_time);
#endif
    std::ostringstream oss;
    oss << std::put_time(&local_time, "%Y%m%d_%H%M%S");
    return oss.str();
}

std::string shell_quote(const std::string &value)
{
    std::string quoted = "'";
    for (char ch : value)
    {
        if (ch == '\'')
        {
            quoted += "'\\''";
        }
        else
        {
            quoted += ch;
        }
    }
    quoted += "'";
    return quoted;
}

std::string file_stem_for_dir(const std::string &path)
{
    size_t slash_pos = path.find_last_of("/\\");
    std::string name = slash_pos == std::string::npos ? path : path.substr(slash_pos + 1);
    size_t dot_pos = name.find_last_of('.');
    if (dot_pos != std::string::npos)
    {
        name = name.substr(0, dot_pos);
    }

    std::string safe;
    for (char ch : name)
    {
        if (std::isalnum(static_cast<unsigned char>(ch)) || ch == '_' || ch == '-')
        {
            safe += ch;
        }
        else
        {
            safe += '_';
        }
    }
    return safe.empty() ? "video" : safe;
}

bool command_succeeds(const std::string &command)
{
    int ret = std::system(command.c_str());
    return ret == 0;
}

bool env_flag_enabled(const char *name)
{
    const char *value = std::getenv(name);
    if (!value)
    {
        return false;
    }

    std::string text(value);
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return text == "1" || text == "true" || text == "yes" || text == "on";
}

double env_double_or_default(const char *name, double default_value, double min_value, double max_value)
{
    const char *value = std::getenv(name);
    if (!value || value[0] == '\0')
    {
        return default_value;
    }

    char *end = nullptr;
    errno = 0;
    const double parsed = std::strtod(value, &end);
    if (errno != 0 || end == value || parsed < min_value || parsed > max_value)
    {
        std::cerr << "warning: ignore invalid " << name << "=" << value
                  << ", expected " << min_value << ".." << max_value << std::endl;
        return default_value;
    }
    return parsed;
}

int system_exit_code(int status)
{
    if (status == -1)
    {
        return -1;
    }
    if (status > 255)
    {
        return (status >> 8) & 0xff;
    }
    return status;
}

void print_current_working_directory()
{
    char cwd[512] = {0};
    if (getcwd(cwd, sizeof(cwd)) != nullptr)
    {
        printf("[video] cwd: %s\n", cwd);
    }
    else
    {
        printf("[video] cwd: failed, errno=%d\n", errno);
    }
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
    const cv::Size text_size = cv::getTextSize(kFallOsdText,
                                               kFontFace,
                                               kFontScale,
                                               kTextThickness,
                                               &baseline);
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
    const cv::Scalar background = has_alpha ? cv::Scalar(0, 0, 180, 220)
                                            : cv::Scalar(0, 0, 180);
    const cv::Scalar text_color = has_alpha ? cv::Scalar(255, 255, 255, 255)
                                            : cv::Scalar(255, 255, 255);
    const cv::Rect label_rect(label_x, label_y, label_w, label_h);

    cv::rectangle(image, label_rect, background, -1, cv::LINE_8, 0);
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
    for (auto r : result.boxes)
    {
        ScopedTiming st("draw boxes", debug_mode);
        int x1 = r.x1 * image.cols / src_width;
        int y1 = r.y1 * image.rows / src_height;
        int w = (r.x2 - r.x1) * image.cols / src_width;
        int h = (r.y2 - r.y1) * image.rows / src_height;
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
    mutable std::mutex mutex;
    std::deque<EvidenceFrame> frame_buffer;
    std::deque<std::chrono::steady_clock::time_point> fall_hit_times;
    bool event_latched = false;
    bool clear_started = false;
    std::chrono::steady_clock::time_point clear_since;
    std::chrono::steady_clock::time_point last_debug_print;
};

double box_width(const FallBox &box)
{
    return std::max(1.0, box.x2 - box.x1);
}

double box_height(const FallBox &box)
{
    return std::max(1.0, box.y2 - box.y1);
}

double box_center_x(const FallBox &box)
{
    return (box.x1 + box.x2) * 0.5;
}

double box_center_y(const FallBox &box)
{
    return (box.y1 + box.y2) * 0.5;
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

void maybe_print_evidence_debug(FallAlarmState &alarm,
                                const FallBox &current_box,
                                std::chrono::steady_clock::time_point now)
{
    if (!env_flag_enabled("FDD_EVIDENCE_DEBUG"))
    {
        return;
    }

    if (alarm.last_debug_print.time_since_epoch().count() != 0 &&
        now - alarm.last_debug_print < std::chrono::seconds(1))
    {
        return;
    }
    alarm.last_debug_print = now;

    std::cout << "[evidence] fall_box=" << (current_box.valid ? 1 : 0)
              << " vote_hits=" << alarm.fall_hit_times.size()
              << "/" << kFallVoteMinHits
              << " window_s=" << kFallVoteWindowSeconds
              << " latched=" << (alarm.event_latched ? 1 : 0)
              << " cached_frames=" << alarm.frame_buffer.size()
              << std::endl;
}

bool encode_evidence_frame(const cv::Mat &frame, std::vector<uchar> &encoded_frame)
{
    encoded_frame.clear();
    if (frame.empty())
    {
        return false;
    }

    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, kEvidenceJpegQuality};
    try
    {
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
    catch (const cv::Exception &ex)
    {
        std::cerr << "warning: failed to encode evidence frame: " << ex.what() << std::endl;
        encoded_frame.clear();
        return false;
    }
}

void cache_evidence_frame(FallAlarmState &alarm,
                          const cv::Mat &frame,
                          std::chrono::steady_clock::time_point now)
{
    if (frame.empty())
    {
        return;
    }

    std::vector<uchar> encoded_frame;
    if (!encode_evidence_frame(frame, encoded_frame))
    {
        return;
    }

    std::lock_guard<std::mutex> lock(alarm.mutex);
    alarm.frame_buffer.push_back({now, encoded_frame});
    const auto oldest_to_keep = now - std::chrono::seconds(kPreEventSeconds);
    while (!alarm.frame_buffer.empty() &&
           alarm.frame_buffer.front().captured_at < oldest_to_keep)
    {
        alarm.frame_buffer.pop_front();
    }
}

std::vector<EvidenceFrame> ordered_cached_frames(const FallAlarmState &alarm)
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

std::string create_unique_event_dir(const std::string &root)
{
    const std::string timestamp = timestamp_string();
    const std::string base = join_path(root, timestamp);

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
        if (errno == EEXIST)
        {
            continue;
        }

        std::cerr << "failed to create fall event directory: " << event_dir << std::endl;
        return "";
    }

    std::cerr << "failed to create unique fall event directory under: " << root << std::endl;
    return "";
}

bool save_frame_sequence(const std::vector<EvidenceFrame> &frames, const std::string &event_dir)
{
    const std::string frames_dir = join_path(event_dir, "frames");
    if (ensure_directory(frames_dir) != 0)
    {
        return false;
    }

    std::cout << "fall evidence frame sequence save started: " << frames_dir
              << " frames=" << frames.size() << std::endl;

    int saved_count = 0;
    for (size_t i = 0; i < frames.size(); ++i)
    {
        if (frames[i].encoded.empty())
        {
            continue;
        }

        char file_name[32] = {0};
        std::snprintf(file_name, sizeof(file_name), "%06d.jpg", static_cast<int>(i + 1));
        const std::string frame_path = join_path(frames_dir, file_name);
        std::ofstream frame_file(frame_path, std::ios::binary);
        if (frame_file)
        {
            frame_file.write(reinterpret_cast<const char *>(frames[i].encoded.data()),
                             static_cast<std::streamsize>(frames[i].encoded.size()));
            frame_file.close();
        }
        if (frame_file.good() && !frame_file.fail())
        {
            ++saved_count;
            if (saved_count % kFrameSaveProgressInterval == 0)
            {
                std::cout << "fall evidence frame sequence progress: "
                          << saved_count << "/" << frames.size() << std::endl;
                usleep(1000);
            }
        }
        else
        {
            std::cerr << "warning: failed to save evidence frame: " << frame_path << std::endl;
        }
    }

    if (saved_count > 0)
    {
        std::cout << "fall evidence frame sequence saved: " << frames_dir
                  << " frames=" << saved_count << std::endl;
        return true;
    }

    std::cerr << "warning: no evidence frames were saved under: " << frames_dir << std::endl;
    return false;
}

double estimate_evidence_fps(const std::vector<EvidenceFrame> &frames, double fallback_fps)
{
    if (frames.size() >= 2)
    {
        const double seconds = std::chrono::duration<double>(frames.back().captured_at -
                                                             frames.front().captured_at)
                                   .count();
        if (seconds > 0.1)
        {
            const double fps = static_cast<double>(frames.size() - 1) / seconds;
            return std::max(1.0, std::min(60.0, fps));
        }
    }
    return fallback_fps > 0.0 ? fallback_fps : kDefaultEvidenceFps;
}

bool save_event_video(const std::vector<EvidenceFrame> &frames,
                      const std::string &event_dir,
                      double fps)
{
    if (frames.empty())
    {
        return false;
    }

    const std::string video_path = join_path(event_dir, "event.avi");
    const double output_fps = estimate_evidence_fps(frames, fps);
    cv::Mat first_frame = cv::imdecode(frames[0].encoded, cv::IMREAD_COLOR);
    if (first_frame.empty())
    {
        std::cerr << "warning: failed to decode first evidence frame for AVI" << std::endl;
        return false;
    }

    cv::Size frame_size(first_frame.cols, first_frame.rows);
    cv::VideoWriter writer;

    try
    {
        writer.open(video_path,
                    cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                    output_fps,
                    frame_size);
        if (!writer.isOpened())
        {
            std::cerr << "warning: failed to create evidence video: " << video_path << std::endl;
            return false;
        }

        int written_count = 0;
        for (const auto &frame : frames)
        {
            if (frame.encoded.empty())
            {
                continue;
            }
            cv::Mat decoded_frame = cv::imdecode(frame.encoded, cv::IMREAD_COLOR);
            if (decoded_frame.empty())
            {
                std::cerr << "warning: skip undecodable evidence frame" << std::endl;
                continue;
            }
            if (decoded_frame.size() != frame_size)
            {
                std::cerr << "warning: skip evidence frame with mismatched size" << std::endl;
                continue;
            }
            writer.write(decoded_frame);
            ++written_count;
        }
        writer.release();

        if (written_count > 0)
        {
            std::cout << "fall evidence video saved: " << video_path
                      << " frames=" << written_count << std::endl;
            return true;
        }

        std::cerr << "warning: evidence video has no frames: " << video_path << std::endl;
        return false;
    }
    catch (const cv::Exception &ex)
    {
        if (writer.isOpened())
        {
            writer.release();
        }
        std::cerr << "warning: exception while saving evidence video: " << ex.what() << std::endl;
        return false;
    }
}

void save_fall_event_worker(const std::vector<EvidenceFrame> &frames, double fps)
{
    const std::string root = prepare_evidence_root();
    if (root.empty())
    {
        std::cerr << "warning: fall evidence root unavailable"
                  << "; set FDD_EVIDENCE_ROOT to the mounted SD card path" << std::endl;
        return;
    }

    const std::string event_dir = create_unique_event_dir(root);
    if (event_dir.empty())
    {
        return;
    }

    const std::string snapshot_path = join_path(event_dir, "snapshot.jpg");
    bool snapshot_saved = false;
    for (auto it = frames.rbegin(); it != frames.rend(); ++it)
    {
        if (it->encoded.empty())
        {
            continue;
        }

        std::ofstream snapshot_file(snapshot_path, std::ios::binary);
        if (snapshot_file)
        {
            snapshot_file.write(reinterpret_cast<const char *>(it->encoded.data()),
                                static_cast<std::streamsize>(it->encoded.size()));
            snapshot_file.close();
        }
        snapshot_saved = snapshot_file.good() && !snapshot_file.fail();
        break;
    }

    if (snapshot_saved)
    {
        std::cout << "fall alarm snapshot saved: " << snapshot_path << std::endl;
    }
    else
    {
        std::cerr << "warning: failed to save fall alarm snapshot: "
                  << snapshot_path << std::endl;
    }

    if (env_flag_enabled("FDD_EVIDENCE_AVI"))
    {
        if (!save_event_video(frames, event_dir, fps))
        {
            save_frame_sequence(frames, event_dir);
        }
        return;
    }

    std::cout << "fall evidence AVI disabled by default; saving JPG frame sequence"
              << std::endl;
    if (!save_frame_sequence(frames, event_dir))
    {
        std::cerr << "warning: failed to save fall evidence frame sequence: "
                  << event_dir << std::endl;
    }
}

void start_fall_event_save(FallAlarmState &alarm, double fps)
{
    std::vector<EvidenceFrame> frames = ordered_cached_frames(alarm);
    if (frames.empty())
    {
        std::cerr << "warning: skip fall evidence save because frame cache is empty" << std::endl;
        return;
    }

    std::cout << "fall evidence save queued: frames=" << frames.size() << std::endl;
    std::thread saver([frames = std::move(frames), fps]() {
        try
        {
            save_fall_event_worker(frames, fps);
        }
        catch (const cv::Exception &ex)
        {
            std::cerr << "warning: fall evidence save failed with OpenCV exception: "
                      << ex.what() << std::endl;
        }
        catch (const std::exception &ex)
        {
            std::cerr << "warning: fall evidence save failed with exception: "
                      << ex.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "warning: fall evidence save failed with unknown exception"
                      << std::endl;
        }
    });
    saver.detach();
}

void update_fall_alarm(FallAlarmState &alarm,
                       const cv::Mat &evidence_frame,
                       const DetectResult &result,
                       double fps)
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
        while (!alarm.fall_hit_times.empty() &&
               alarm.fall_hit_times.front() < vote_window_start)
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
            if (alarm.event_latched &&
                now - alarm.clear_since >= std::chrono::seconds(kEventResetSeconds))
            {
                alarm.event_latched = false;
                alarm.fall_hit_times.clear();
            }
            maybe_print_evidence_debug(alarm, current_box, now);
            return;
        }

        alarm.clear_started = false;
        alarm.fall_hit_times.push_back(now);
        if (!alarm.event_latched && alarm.fall_hit_times.size() >= kFallVoteMinHits)
        {
            alarm.event_latched = true;
            trigger_save = true;
        }
        maybe_print_evidence_debug(alarm, current_box, now);
    }
    if (trigger_save)
    {
        start_fall_event_save(alarm, fps);
    }
}
} // namespace

static void draw_detect_boxes(cv::Mat &image, const DetectResult &result, int debug_mode)
{
    draw_boxes_scaled(image, result, image.cols, image.rows, debug_mode, cv::Scalar(0, 0, 255));
}

static bool local_display_needs_rotate()
{
#if defined(CONFIG_BOARD_K230D_CANMV) || defined(CONFIG_BOARD_K230_CANMV_DONGSHANPI) || defined(CONFIG_BOARD_K230_CANMV_01STUDIO)
    return true;
#else
    return false;
#endif
}

class LocalVoDisplay
{
public:
    explicit LocalVoDisplay(bool rotate_display) : rotate_display_(rotate_display)
    {
        if (vo_only_start() != 0)
        {
            std::cerr << "warning: failed to initialize local VO display" << std::endl;
            return;
        }

        memset(&vf_info_, 0, sizeof(vf_info_));
        vf_info_.v_frame.width = osd_width;
        vf_info_.v_frame.height = osd_height;
        vf_info_.v_frame.stride[0] = osd_width;
        vf_info_.v_frame.pixel_format = PIXEL_FORMAT_ARGB_8888;
        block = vo_insert_frame(&vf_info_, &pic_vaddr_);
        if (block == VB_INVALID_HANDLE || pic_vaddr_ == NULL)
        {
            std::cerr << "warning: failed to allocate local VO frame" << std::endl;
            if (vicap_install_osd == 1)
            {
                kd_mpi_vo_osd_disable(osd_id);
            }
            vo_only_stop();
            return;
        }
        ready_ = true;
    }

    ~LocalVoDisplay()
    {
        if (ready_)
        {
            vo_osd_release_block();
            vo_only_stop();
        }
    }

    bool ready() const
    {
        return ready_;
    }

    void show(const cv::Mat &bgr_frame)
    {
        if (!ready_ || bgr_frame.empty())
        {
            return;
        }

        cv::Mat display_src;
        if (rotate_display_)
        {
            cv::rotate(bgr_frame, display_src, cv::ROTATE_90_CLOCKWISE);
        }
        else
        {
            display_src = bgr_frame;
        }

        cv::Mat resized;
        cv::resize(display_src, resized, cv::Size(osd_width, osd_height));
        cv::Mat argb_frame;
        cv::cvtColor(resized, argb_frame, cv::COLOR_BGR2BGRA);
        memcpy(pic_vaddr_, argb_frame.data, osd_width * osd_height * 4);
        kd_mpi_vo_chn_insert_frame(osd_id + 3, &vf_info_);
    }

private:
    bool ready_ = false;
    bool rotate_display_ = false;
    k_video_frame_info vf_info_;
    void *pic_vaddr_ = NULL;
};

static int process_image_file(const char *kmodel_path, const char *image_path, int debug_mode)
{
    cv::Mat ori_img = cv::imread(image_path);
    if (ori_img.empty())
    {
        std::cerr << "failed to read image: " << image_path << std::endl;
        return -1;
    }

    int ori_w = ori_img.cols;
    int ori_h = ori_img.rows;

    std::vector<uint8_t> chw_vec;
    Utils::hwc_to_chw(ori_img, chw_vec);

    AiContext ai_ctx = {};
    if (ai_init(&ai_ctx, kmodel_path, ori_img.channels(), ori_h, ori_w) != 0)
    {
        return -1;
    }

    DetectResult result = ai_run_frame(&ai_ctx, reinterpret_cast<uintptr_t>(chw_vec.data()), 0);
    draw_detect_boxes(ori_img, result, debug_mode);

    ai_deinit(&ai_ctx);
    cv::imwrite("fdd_result.jpg", ori_img);

    LocalVoDisplay display(local_display_needs_rotate());
    if (display.ready())
    {
        display.show(ori_img);
        std::cout << "image displayed, press q to exit" << std::endl;
        while (getchar() != 'q')
        {
            usleep(10000);
        }
    }
    return 0;
}

static int process_frame_sequence(const char *kmodel_path, const char *frames_dir, int debug_mode)
{
    std::vector<std::string> frame_paths = list_frame_images(frames_dir);
    if (frame_paths.empty())
    {
        std::cerr << "no jpg/png/bmp frames found in: " << frames_dir << std::endl;
        return -1;
    }

    printf("[frames] input: %s count=%d\n", frames_dir, static_cast<int>(frame_paths.size()));

    cv::Mat first_frame = cv::imread(frame_paths[0]);
    if (first_frame.empty())
    {
        std::cerr << "failed to read first frame: " << frame_paths[0] << std::endl;
        return -1;
    }

    const int frame_w = first_frame.cols;
    const int frame_h = first_frame.rows;
    const double fps = kDefaultEvidenceFps;
    printf("[frames] first frame: %dx%d channels=%d\n", frame_w, frame_h, first_frame.channels());

    cv::VideoWriter writer;
    const std::string output_path = "fdd_video_result.avi";
    const bool request_result_video = env_flag_enabled("FDD_SAVE_RESULT_VIDEO");
    bool save_result_video = false;
    if (request_result_video)
    {
        writer.open(output_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(frame_w, frame_h));
        save_result_video = writer.isOpened();
    }
    else
    {
        printf("[frames] full result video disabled; set FDD_SAVE_RESULT_VIDEO=1 to enable %s\n", output_path.c_str());
    }
    if (!save_result_video)
    {
        printf("[frames] continue display/inference without full result video\n");
    }

    AiContext ai_ctx = {};
    if (ai_init(&ai_ctx, kmodel_path, first_frame.channels(), frame_h, frame_w) != 0)
    {
        return -1;
    }

    FallAlarmState alarm;
    LocalVoDisplay display(local_display_needs_rotate());
    const int frame_interval_us = static_cast<int>(1000000.0 / fps);

    for (size_t frame_index = 0; frame_index < frame_paths.size(); ++frame_index)
    {
        auto frame_start = std::chrono::steady_clock::now();
        ScopedTiming st("frame sequence", debug_mode);

        cv::Mat frame = cv::imread(frame_paths[frame_index]);
        if (frame.empty())
        {
            std::cerr << "skip unreadable frame: " << frame_paths[frame_index] << std::endl;
            continue;
        }
        if (frame.cols != frame_w || frame.rows != frame_h || frame.channels() != ai_ctx.channel)
        {
            std::cerr << "skip unsupported frame shape: " << frame_paths[frame_index] << std::endl;
            continue;
        }

        std::vector<uint8_t> chw_vec;
        Utils::hwc_to_chw(frame, chw_vec);
        DetectResult result = ai_run_frame(&ai_ctx, reinterpret_cast<uintptr_t>(chw_vec.data()), 0);
        draw_detect_boxes(frame, result, debug_mode);
        if (save_result_video)
        {
            writer.write(frame);
        }
        update_fall_alarm(alarm, frame, result, fps);
        display.show(frame);

        auto frame_stop = std::chrono::steady_clock::now();
        int elapsed_us = static_cast<int>(std::chrono::duration<double, std::micro>(frame_stop - frame_start).count());
        if (frame_interval_us > elapsed_us)
        {
            usleep(frame_interval_us - elapsed_us);
        }
    }

    ai_deinit(&ai_ctx);
    if (save_result_video)
    {
        std::cout << "frame sequence result saved to " << output_path << std::endl;
    }
    else
    {
        std::cout << "frame sequence processing finished without result video file" << std::endl;
    }
    return 0;
}
static int process_video_file(const char *kmodel_path, const char *video_path, int debug_mode)
{
    printf("[video] input: %s\n", video_path);
    printf("[video] mode: auto ffmpeg preprocess\n");
    print_current_working_directory();

    struct stat video_stat;
    if (stat(video_path, &video_stat) != 0)
    {
        printf("[video] stat: failed, errno=%d\n", errno);
        std::cerr << "[video] error: video file not found: " << video_path << std::endl;
        std::cerr << "[video] suggestion: check the file path and current working directory." << std::endl;
        return -1;
    }
    printf("[video] stat: exists, size=%lld bytes\n", static_cast<long long>(video_stat.st_size));

    const std::string ffmpeg_check = "ffmpeg -version > /dev/null 2>&1";
    const bool ffmpeg_available = command_succeeds(ffmpeg_check);
    printf("[video] ffmpeg: %s\n", ffmpeg_available ? "found" : "not found");
    if (!ffmpeg_available)
    {
        std::cerr << "[video] error: ffmpeg not found or not executable from PATH." << std::endl;
        std::cerr << "[video] suggestion: put ffmpeg on the board PATH, or pass a prepared frame directory." << std::endl;
        return -1;
    }

    const std::string frames_dir = std::string("fdd_frames_") + file_stem_for_dir(video_path) + "_" + timestamp_string();
    if (ensure_directory(frames_dir) != 0)
    {
        std::cerr << "[video] error: failed to create frame directory: " << frames_dir << std::endl;
        return -1;
    }

    const std::string ffmpeg_log = frames_dir + "/ffmpeg.log";
    const std::string frame_pattern = frames_dir + "/%06d.jpg";
    std::ostringstream command;
    command << "ffmpeg -y -i " << shell_quote(video_path)
            << " -vf fps=" << static_cast<int>(kDefaultEvidenceFps)
            << " -q:v 3 "
            << shell_quote(frame_pattern)
            << " > " << shell_quote(ffmpeg_log)
            << " 2>&1";

    const std::string ffmpeg_command = command.str();
    printf("[video] frame dir: %s\n", frames_dir.c_str());
    printf("[video] ffmpeg command: %s\n", ffmpeg_command.c_str());
    printf("[video] ffmpeg log: %s\n", ffmpeg_log.c_str());

    int ffmpeg_status = std::system(ffmpeg_command.c_str());
    int ffmpeg_ret = system_exit_code(ffmpeg_status);
    printf("[video] ffmpeg exit code: %d\n", ffmpeg_ret);

    std::vector<std::string> generated_frames = list_frame_images(frames_dir);
    printf("[video] generated frames: %d\n", static_cast<int>(generated_frames.size()));
    if (ffmpeg_ret != 0)
    {
        std::cerr << "[video] error: ffmpeg failed while preprocessing the video." << std::endl;
        std::cerr << "[video] suggestion: open " << ffmpeg_log
                  << " and check whether the video codec/container is supported or the file is damaged." << std::endl;
        return -1;
    }
    if (generated_frames.empty())
    {
        std::cerr << "[video] error: ffmpeg finished but generated 0 frames." << std::endl;
        std::cerr << "[video] suggestion: inspect " << ffmpeg_log
                  << " and confirm the input video has readable frames." << std::endl;
        return -1;
    }

    cv::Mat first_frame = cv::imread(generated_frames[0]);
    if (first_frame.empty())
    {
        std::cerr << "[video] error: generated first frame cannot be read by OpenCV imread: "
                  << generated_frames[0] << std::endl;
        std::cerr << "[video] suggestion: check generated jpg files in " << frames_dir << "." << std::endl;
        return -1;
    }
    printf("[video] first generated frame: %dx%d channels=%d\n", first_frame.cols, first_frame.rows, first_frame.channels());

    return process_frame_sequence(kmodel_path, frames_dir.c_str(), debug_mode);
}

namespace
{
struct SharedIspFrame
{
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<uint8_t> latest_chw;
    uint64_t sequence = 0;
    bool stopped = false;
};

struct SharedDetectionResult
{
    std::mutex mutex;
    DetectResult result;
};

void publish_latest_isp_frame(SharedIspFrame &shared_frame,
                              const std::vector<uint8_t> &frame_chw)
{
    {
        std::lock_guard<std::mutex> lock(shared_frame.mutex);
        shared_frame.latest_chw = frame_chw;
        ++shared_frame.sequence;
    }
    shared_frame.cv.notify_one();
}

bool wait_for_latest_isp_frame(SharedIspFrame &shared_frame,
                               uint64_t &last_sequence,
                               std::vector<uint8_t> &frame_chw)
{
    std::unique_lock<std::mutex> lock(shared_frame.mutex);
    shared_frame.cv.wait_for(lock, std::chrono::milliseconds(100), [&]() {
        return shared_frame.sequence != last_sequence ||
               shared_frame.stopped ||
               isp_stop.load();
    });

    if (shared_frame.sequence == last_sequence || shared_frame.latest_chw.empty())
    {
        return false;
    }

    frame_chw = shared_frame.latest_chw;
    last_sequence = shared_frame.sequence;
    return true;
}

void stop_shared_isp_frame(SharedIspFrame &shared_frame)
{
    {
        std::lock_guard<std::mutex> lock(shared_frame.mutex);
        shared_frame.stopped = true;
    }
    shared_frame.cv.notify_all();
}

void update_shared_detection(SharedDetectionResult &shared_detection,
                             const DetectResult &result)
{
    std::lock_guard<std::mutex> lock(shared_detection.mutex);
    shared_detection.result = result;
}

DetectResult latest_shared_detection(SharedDetectionResult &shared_detection)
{
    std::lock_guard<std::mutex> lock(shared_detection.mutex);
    return shared_detection.result;
}

void isp_capture_loop(FallAlarmState &alarm,
                      SharedIspFrame &shared_frame,
                      SharedDetectionResult &shared_detection,
                      size_t frame_size,
                      int debug_mode,
                      double evidence_fps)
{
    std::vector<uint8_t> frame_chw(frame_size);
    auto last_evidence_cache = std::chrono::steady_clock::time_point{};
    const auto evidence_interval = std::chrono::duration<double>(1.0 / evidence_fps);

    while (!isp_stop)
    {
        const auto loop_start = std::chrono::steady_clock::now();
        k_video_frame_info local_dump_info;
        memset(&local_dump_info, 0, sizeof(k_video_frame_info));

        {
            ScopedTiming st("read capture", debug_mode);
            int ret = kd_mpi_vicap_dump_frame(vicap_dev, VICAP_CHN_ID_1, VICAP_DUMP_YUV, &local_dump_info, 1000);
            if (ret)
            {
                printf("sample_vicap...kd_mpi_vicap_dump_frame failed.\n");
                continue;
            }
        }

        bool copied = false;
        {
            ScopedTiming st("isp copy", debug_mode);
            auto vbvaddr = kd_mpi_sys_mmap_cached(local_dump_info.v_frame.phys_addr[0], frame_size);
            if (vbvaddr != nullptr)
            {
                memcpy(frame_chw.data(), (void *)vbvaddr, frame_size);
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

        if (!copied)
        {
            continue;
        }

        publish_latest_isp_frame(shared_frame, frame_chw);

        const auto now = std::chrono::steady_clock::now();
        if (last_evidence_cache.time_since_epoch().count() == 0 ||
            now - last_evidence_cache >= evidence_interval)
        {
            DetectResult result = latest_shared_detection(shared_detection);
            cv::Mat evidence_frame = rgb_chw_to_bgr_mat(frame_chw.data(), SENSOR_WIDTH, SENSOR_HEIGHT);
            draw_boxes_scaled(evidence_frame,
                              result,
                              SENSOR_WIDTH,
                              SENSOR_HEIGHT,
                              debug_mode,
                              cv::Scalar(0, 0, 255));
            cache_evidence_frame(alarm, evidence_frame, now);
            last_evidence_cache = now;
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

    stop_shared_isp_frame(shared_frame);
}

} // namespace

void video_proc_common(char *argv[], bool rotate_osd_for_draw)
{
    const int debug_mode = atoi(argv[3]);
    const double evidence_fps = env_double_or_default("FDD_ISP_EVIDENCE_FPS",
                                                     kDefaultIspEvidenceFps,
                                                     1.0,
                                                     15.0);
    std::cout << "fall evidence ISP cache fps: " << evidence_fps << std::endl;
    vivcap_start();

    k_video_frame_info vf_info;
    void *pic_vaddr = NULL;
    bool display_ready = FDD_DISPLAY_IS_ENABLED();

    memset(&vf_info, 0, sizeof(vf_info));

    if (display_ready)
    {
        vf_info.v_frame.width = osd_width;
        vf_info.v_frame.height = osd_height;
        vf_info.v_frame.stride[0] = osd_width;
        vf_info.v_frame.pixel_format = PIXEL_FORMAT_ARGB_8888;
        block = vo_insert_frame(&vf_info, &pic_vaddr);
        if (block == VB_INVALID_HANDLE || pic_vaddr == NULL)
        {
            std::cerr << "warning: failed to allocate OSD frame, continue headless" << std::endl;
            display_ready = false;
        }
    }

    size_t size = SENSOR_CHANNEL * SENSOR_HEIGHT * SENSOR_WIDTH;

    AiContext ai_ctx = {};
    if (ai_init(&ai_ctx, argv[1], SENSOR_CHANNEL, SENSOR_HEIGHT, SENSOR_WIDTH) != 0)
    {
        std::abort();
    }

    FallAlarmState alarm;
    SharedIspFrame shared_frame;
    SharedDetectionResult shared_detection;
    DetectResult result;
    std::vector<uint8_t> ai_frame;
    uint64_t last_sequence = 0;

    std::thread capture_thread(isp_capture_loop,
                               std::ref(alarm),
                               std::ref(shared_frame),
                               std::ref(shared_detection),
                               size,
                               debug_mode,
                               evidence_fps);

    while (!isp_stop)
    {
        if (!wait_for_latest_isp_frame(shared_frame, last_sequence, ai_frame))
        {
            continue;
        }

        ScopedTiming st("total time", 1);

        result = ai_run_frame(&ai_ctx, reinterpret_cast<uintptr_t>(ai_frame.data()), 0);
        update_shared_detection(shared_detection, result);
        cv::Mat trigger_evidence_frame;
        if (!result.boxes.empty())
        {
            trigger_evidence_frame = rgb_chw_to_bgr_mat(ai_frame.data(), SENSOR_WIDTH, SENSOR_HEIGHT);
            draw_boxes_scaled(trigger_evidence_frame,
                              result,
                              SENSOR_WIDTH,
                              SENSOR_HEIGHT,
                              debug_mode,
                              cv::Scalar(0, 0, 255));
        }
        update_fall_alarm(alarm, trigger_evidence_frame, result, evidence_fps);

        if (display_ready)
        {
            cv::Mat osd_frame(osd_height, osd_width, CV_8UC4, cv::Scalar(0, 0, 0, 0));
            cv::Mat draw_frame = osd_frame;
            if (rotate_osd_for_draw)
            {
                cv::rotate(osd_frame, draw_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
            }

            draw_boxes_scaled(draw_frame,
                              result,
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
                memcpy(pic_vaddr, osd_frame.data, osd_width * osd_height * 4);
                kd_mpi_vo_chn_insert_frame(osd_id + 3, &vf_info);
            }
        }
    }

    stop_shared_isp_frame(shared_frame);
    if (capture_thread.joinable())
    {
        capture_thread.join();
    }

    ai_deinit(&ai_ctx);
    if (display_ready)
    {
        vo_osd_release_block();
    }
    vivcap_stop();
}
int main(int argc, char *argv[])
{
    std::cout << "case " << argv[0] << " built at " << __DATE__ << " " << __TIME__ << std::endl;
    if (argc != 4)
    {
        print_usage(argv[0]);
        return -1;
    }

    if (strcmp(argv[2], "None") == 0)
    {
        #if defined(CONFIG_BOARD_K230_CANMV)
        {
            std::thread thread_isp(video_proc_common, argv, false);
            while (getchar() != 'q')
            {
                usleep(10000);
            }

            isp_stop = true;
            thread_isp.join();
        }
        #elif defined(CONFIG_BOARD_K230_CANMV_V2)
        {
            std::thread thread_isp(video_proc_common, argv, false);
            while (getchar() != 'q')
            {
                usleep(10000);
            }

            isp_stop = true;
            thread_isp.join();
        }
        #elif defined(CONFIG_BOARD_K230D_CANMV)
        {
            std::thread thread_isp(video_proc_common, argv, true);
            while (getchar() != 'q')
            {
                usleep(10000);
            }

            isp_stop = true;
            thread_isp.join();
        }
        #elif defined(CONFIG_BOARD_K230_CANMV_DONGSHANPI)
        {
            std::thread thread_isp(video_proc_common, argv, true);
            while (getchar() != 'q')
            {
                usleep(10000);
            }

            isp_stop = true;
            thread_isp.join();
        }
	#elif defined(CONFIG_BOARD_K230_CANMV_01STUDIO)
        {
            std::thread thread_isp(video_proc_common, argv, true);
            while (getchar() != 'q')
            {
                usleep(10000);
            }

            isp_stop = true;
            thread_isp.join();
        }
        #else
        {
            std::thread thread_isp(video_proc_common, argv, false);
            while (getchar() != 'q')
            {
                usleep(10000);
            }

            isp_stop = true;
            thread_isp.join();
        }
        #endif
    }
    else if (is_directory_path(argv[2]))
    {
        return process_frame_sequence(argv[1], argv[2], atoi(argv[3]));
    }
    else if (has_video_extension(argv[2]))
    {
        return process_video_file(argv[1], argv[2], atoi(argv[3]));
    }
    else
    {
        return process_image_file(argv[1], argv[2], atoi(argv[3]));
    }
    return 0;
}
