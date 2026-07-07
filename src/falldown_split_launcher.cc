#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <unistd.h>

namespace
{
void print_usage(const char *name)
{
    std::cout << "Usage: " << name << " <kmodel> <debug_mode> [display_mode]" << std::endl
              << "For example:" << std::endl
              << "  ./" << name << " best_fp16.kmodel 0" << std::endl
              << "  ./" << name << " best_fp16.kmodel 0 off" << std::endl
              << "display_mode: auto | on | off" << std::endl;
}

bool valid_display_mode(const char *value)
{
    return value != nullptr &&
           (std::strcmp(value, "auto") == 0 ||
            std::strcmp(value, "on") == 0 ||
            std::strcmp(value, "off") == 0);
}

std::string executable_dir(const char *argv0)
{
    const std::string path = argv0 ? argv0 : "";
    const size_t slash = path.find_last_of('/');
    if (slash == std::string::npos)
    {
        return ".";
    }
    if (slash == 0)
    {
        return "/";
    }
    return path.substr(0, slash);
}

std::string join_path(const std::string &dir, const char *name)
{
    if (dir.empty() || dir == ".")
    {
        return std::string("./") + name;
    }
    if (dir.back() == '/')
    {
        return dir + name;
    }
    return dir + "/" + name;
}
}

int main(int argc, char *argv[])
{
    std::cout << "falldown_split_launcher built at " << __DATE__ << " " << __TIME__ << std::endl;
    if (argc != 3 && argc != 4)
    {
        print_usage(argv[0]);
        return -1;
    }

    if (argc == 4)
    {
        if (!valid_display_mode(argv[3]))
        {
            std::cerr << "[launcher] invalid display_mode: " << argv[3] << std::endl;
            print_usage(argv[0]);
            return -1;
        }
        setenv("FDD_DISPLAY", argv[3], 1);
        std::cout << "[launcher] FDD_DISPLAY=" << argv[3] << std::endl;
    }

    const std::string bin_dir = executable_dir(argv[0]);
    const std::string ai_path = join_path(bin_dir, "falldown_ai_event.elf");
    const std::string video_path = join_path(bin_dir, "falldown_video_display.elf");

    pid_t ai_pid = fork();
    if (ai_pid < 0)
    {
        std::cerr << "[launcher] fork AI process failed: " << std::strerror(errno) << std::endl;
        return -1;
    }

    if (ai_pid == 0)
    {
        execl(ai_path.c_str(), ai_path.c_str(), argv[1], argv[2], static_cast<char *>(nullptr));
        std::cerr << "[launcher] exec AI process failed: " << std::strerror(errno) << std::endl;
        _exit(127);
    }

    std::cout << "[launcher] AI process pid=" << ai_pid << std::endl;
    sleep(1);

    std::cout << "[launcher] exec video process in foreground" << std::endl;
    execl(video_path.c_str(), video_path.c_str(), argv[2], static_cast<char *>(nullptr));

    std::cerr << "[launcher] exec video process failed: " << std::strerror(errno) << std::endl;
    kill(ai_pid, SIGTERM);
    return -1;
}
