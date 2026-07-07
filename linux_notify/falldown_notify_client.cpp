#include <atomic>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>
#include <ctime>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>

#include "comm_client.h"

namespace
{
std::atomic<bool> g_exit{false};
const char *kDefaultLogPath = "/sharefs/sdcard/fall_events.log";

void handle_signal(int)
{
    g_exit.store(true);
}

std::string now_iso8601()
{
    std::time_t now = std::time(nullptr);
    std::tm tm_now;
    localtime_r(&now, &tm_now);

    char buffer[64] = {0};
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S%z", &tm_now);
    return buffer;
}

bool starts_with(const std::string &value, const std::string &prefix)
{
    return value.rfind(prefix, 0) == 0;
}

void ensure_parent_dir(const std::string &path)
{
    const size_t slash = path.find_last_of('/');
    if (slash == std::string::npos || slash == 0)
    {
        return;
    }

    std::string dir = path.substr(0, slash);
    size_t pos = 1;
    while ((pos = dir.find('/', pos)) != std::string::npos)
    {
        const std::string partial = dir.substr(0, pos);
        if (!partial.empty())
        {
            mkdir(partial.c_str(), 0755);
        }
        ++pos;
    }
    mkdir(dir.c_str(), 0755);
}

struct HttpUrl
{
    std::string host;
    std::string port = "80";
    std::string path = "/";
};

bool parse_http_url(const std::string &url, HttpUrl &out)
{
    const std::string prefix = "http://";
    if (!starts_with(url, prefix))
    {
        return false;
    }

    std::string rest = url.substr(prefix.size());
    const size_t path_pos = rest.find('/');
    std::string host_port = path_pos == std::string::npos ? rest : rest.substr(0, path_pos);
    out.path = path_pos == std::string::npos ? "/" : rest.substr(path_pos);

    const size_t colon = host_port.find(':');
    if (colon == std::string::npos)
    {
        out.host = host_port;
        out.port = "80";
    }
    else
    {
        out.host = host_port.substr(0, colon);
        out.port = host_port.substr(colon + 1);
    }

    return !out.host.empty() && !out.port.empty();
}

bool send_all(int fd, const std::string &data)
{
    const char *ptr = data.data();
    size_t remaining = data.size();
    while (remaining > 0)
    {
        const ssize_t sent = send(fd, ptr, remaining, 0);
        if (sent <= 0)
        {
            return false;
        }
        ptr += sent;
        remaining -= static_cast<size_t>(sent);
    }
    return true;
}

bool http_post_json(const std::string &url, const std::string &payload, std::string &error)
{
    HttpUrl parsed;
    if (!parse_http_url(url, parsed))
    {
        error = "only http://host[:port]/path is supported";
        return false;
    }

    addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo *result = nullptr;
    const int gai = getaddrinfo(parsed.host.c_str(), parsed.port.c_str(), &hints, &result);
    if (gai != 0)
    {
        error = gai_strerror(gai);
        return false;
    }

    int fd = -1;
    for (addrinfo *rp = result; rp != nullptr; rp = rp->ai_next)
    {
        fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (fd < 0)
        {
            continue;
        }

        timeval timeout;
        timeout.tv_sec = 2;
        timeout.tv_usec = 0;
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

        if (connect(fd, rp->ai_addr, rp->ai_addrlen) == 0)
        {
            break;
        }
        close(fd);
        fd = -1;
    }
    freeaddrinfo(result);

    if (fd < 0)
    {
        error = "connect failed";
        return false;
    }

    std::ostringstream request;
    request << "POST " << parsed.path << " HTTP/1.1\r\n"
            << "Host: " << parsed.host << "\r\n"
            << "User-Agent: falldown_notify_client/1.0\r\n"
            << "Content-Type: application/json\r\n"
            << "Content-Length: " << payload.size() << "\r\n"
            << "Connection: close\r\n\r\n"
            << payload;

    if (!send_all(fd, request.str()))
    {
        close(fd);
        error = "send failed";
        return false;
    }

    char response[256] = {0};
    const ssize_t received = recv(fd, response, sizeof(response) - 1, 0);
    close(fd);
    if (received <= 0)
    {
        error = "no response";
        return false;
    }

    std::string status(response, response + received);
    if (status.find("HTTP/1.1 2") == 0 || status.find("HTTP/1.0 2") == 0)
    {
        return true;
    }

    const size_t line_end = status.find("\r\n");
    error = line_end == std::string::npos ? status : status.substr(0, line_end);
    return false;
}

class FallNotifyClient : public IClientCallback
{
public:
    FallNotifyClient(std::string log_path, std::string notify_url)
        : log_path_(std::move(log_path)), notify_url_(std::move(notify_url))
    {
        if (!log_path_.empty() && log_path_ != "-")
        {
            ensure_parent_dir(log_path_);
        }
    }

    void OnMessage(const UserMessage *message) override
    {
        if (!message)
        {
            return;
        }

        UserMsgType type = static_cast<UserMsgType>(message->type);
        if (type == UserMsgType::SERVER_MSG_BASE && message->len > 0)
        {
            std::string payload(message->data, message->data + message->len);
            std::cout << "[fall-notify] event: " << payload << std::endl;
            AppendEventLog(payload);
            PushEventAsync(payload);
            return;
        }

        std::cout << "[fall-notify] message type=" << message->type
                  << " len=" << message->len << std::endl;
    }

    void OnEvent(const UserEventData &event) override
    {
        std::cout << "[fall-notify] event fifo type=" << static_cast<int>(event.type)
                  << " timestamp_ms=" << event.timestamp_ms
                  << " jpeg_size=" << event.jpeg_size << std::endl;
    }

    void OnVEncFrameData(const AVEncFrameData &data) override
    {
        std::cout << "[fall-notify] video frame size=" << data.size
                  << " sequence=" << data.sequence << std::endl;
    }

    void OnAEncFrameData(const AVEncFrameData &data) override
    {
        std::cout << "[fall-notify] audio frame size=" << data.size
                  << " sequence=" << data.sequence << std::endl;
    }

private:
    void AppendEventLog(const std::string &payload)
    {
        if (log_path_.empty() || log_path_ == "-")
        {
            return;
        }

        std::lock_guard<std::mutex> lock(log_mutex_);
        std::ofstream file(log_path_, std::ios::app);
        if (!file)
        {
            std::cerr << "[fall-notify] failed to open log: " << log_path_ << std::endl;
            return;
        }

        file << "{\"received_at\":\"" << now_iso8601() << "\",\"event\":" << payload << "}" << std::endl;
        std::cout << "[fall-notify] event appended to " << log_path_ << std::endl;
    }

    void PushEventAsync(const std::string &payload)
    {
        if (notify_url_.empty() || notify_url_ == "-")
        {
            return;
        }

        const std::string url = notify_url_;
        std::thread([url, payload]() {
            std::string error;
            if (http_post_json(url, payload, error))
            {
                std::cout << "[fall-notify] HTTP push ok: " << url << std::endl;
            }
            else
            {
                std::cerr << "[fall-notify] HTTP push failed: " << error << std::endl;
            }
        }).detach();
    }

private:
    std::string log_path_;
    std::string notify_url_;
    std::mutex log_mutex_;
};

void print_usage(const char *name)
{
    std::cout << "Usage: " << name << " [log_path|-] [http_url|-]" << std::endl
              << "Examples:" << std::endl
              << "  ./" << name << std::endl
              << "  ./" << name << " /sharefs/sdcard/fall_events.log" << std::endl
              << "  ./" << name << " /sharefs/sdcard/fall_events.log http://192.168.1.100:8080/api/fall-events" << std::endl
              << "Environment:" << std::endl
              << "  FALL_NOTIFY_LOG=/sharefs/sdcard/fall_events.log" << std::endl
              << "  FALL_NOTIFY_URL=http://host:port/api/fall-events" << std::endl;
}
}

int main(int argc, char *argv[])
{
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
    std::signal(SIGPIPE, SIG_IGN);

    if (argc > 1 && (std::strcmp(argv[1], "-h") == 0 || std::strcmp(argv[1], "--help") == 0))
    {
        print_usage(argv[0]);
        return 0;
    }

    std::string log_path = std::getenv("FALL_NOTIFY_LOG") ? std::getenv("FALL_NOTIFY_LOG") : kDefaultLogPath;
    std::string notify_url = std::getenv("FALL_NOTIFY_URL") ? std::getenv("FALL_NOTIFY_URL") : "";

    if (argc > 1)
    {
        std::string arg1 = argv[1];
        if (starts_with(arg1, "http://"))
        {
            notify_url = arg1;
        }
        else
        {
            log_path = arg1;
        }
    }
    if (argc > 2)
    {
        notify_url = argv[2];
    }

    FallNotifyClient callback(log_path, notify_url);
    UserCommClient client;
    if (client.Init("falldown", &callback) < 0)
    {
        std::cerr << "[fall-notify] failed to init comm client; start RT-Smart video process first" << std::endl;
        return -1;
    }
    if (client.Start() < 0)
    {
        std::cerr << "[fall-notify] failed to send CLIENT_READY" << std::endl;
        client.DeInit();
        return -1;
    }

    std::cout << "[fall-notify] connected; waiting for fall events" << std::endl
              << "[fall-notify] log_path=" << log_path << std::endl
              << "[fall-notify] http_url=" << (notify_url.empty() ? "(disabled)" : notify_url) << std::endl;
    while (!g_exit.load())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    client.Stop();
    client.DeInit();
    std::cout << "[fall-notify] stopped" << std::endl;
    return 0;
}
