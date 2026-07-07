# Falldown AI Inference

这是一个基于 nncase/KPU 的 K230 跌倒检测示例工程。程序支持单张图片推理、板端视频预处理、帧序列视频测试和实时 ISP 摄像头推理，并可在输出图片或板端 OSD 上绘制检测到的跌倒人员框。

## 项目结构

```text
.
|-- CMakeLists.txt
|-- include/                 # 当前构建使用的公共头文件
|   |-- ai_inference.h
|   |-- vi_vo.h
|   `-- scoped_timing.hpp
|-- src/                     # 当前启用的应用和推理实现
|   |-- main.cc
|   `-- ai_inference.cc
|-- models/                  # 训练或导出的模型文件
|   |-- best.pt
|   `-- best_fp16.onnx
|-- tools/
|   `-- onnx_test/           # Windows ONNXRuntime 批量测试流程
|   `-- video_prepare/       # 主机侧视频转帧工具
|-- legacy/
|   |-- split_sources/       # 较早的分体源码实现，当前不参与构建
|   `-- model_duplicates/    # 作为备份保留的重复模型文件
```

当前构建路径有意只使用 `src/main.cc` 和 `src/ai_inference.cc`。`legacy/split_sources/` 下的文件仅作为历史参考副本保留，避免未启用源码与当前构建路径混在一起。

## 使用方式

```text
Usage: ./falldown_detect.elf <kmodel> <input_mode> <debug_mode>
```

示例：

```sh
# 图片模式
./falldown_detect.elf yolov5n-falldown.kmodel falldown_elder.jpg 0

# 视频模式，板端自动使用 ffmpeg 预处理
./falldown_detect.elf yolov5n-falldown.kmodel falldown.avi 0

# 帧序列模式，可作为回退或调试输入
./falldown_detect.elf yolov5n-falldown.kmodel frames 0

# ISP 摄像头模式
./falldown_detect.elf yolov5n-falldown.kmodel None 0
```

参数说明：

- `kmodel`：K230 `.kmodel` 文件路径。
- `input_mode`：图片路径、视频路径、帧目录；传入 `None` 时进入 ISP 摄像头模式。
- `debug_mode`：`0` 关闭性能打印，`1` 打印简单耗时，`2` 打印更详细的信息。

部署版本使用 `src/ai_inference.cc` 中的固定阈值：`kDefaultObjThreshold = 0.5f` 和 `kDefaultNmsThreshold = 0.45f`。如需修改阈值，请直接修改这些常量并重新编译。部署命令中有意不再提供运行时阈值参数，以减少板端运行参数错误。

图片模式会在当前工作目录写出 `fdd_result.jpg`。如果 VO 初始化可用，程序还会把检测结果全屏显示到板端显示屏或 HDMI，直到按下 `q` 退出。

K230 板端视频测试时，可以直接传入原始视频文件。程序会使用板端 `ffmpeg` 将视频转换为临时的补零编号帧目录，然后复用帧序列推理路径：

```sh
./falldown_detect.elf best_fp16.kmodel falldown.avi 0
```

视频模式会打印清晰的诊断信息，包括文件 stat、当前工作目录、ffmpeg 可用性、完整 ffmpeg 命令、ffmpeg 日志路径、退出码和生成帧数量。临时帧目录会保留为 `fdd_frames_<video>_<timestamp>/`，便于排查问题。

帧序列模式仍作为可选的回退和调试输入保留：

```sh
./falldown_detect.elf best_fp16.kmodel frames 0
```

`tools/video_prepare/` 中的电脑端辅助工具仍可手动准备帧目录，但正常板端测试建议优先使用上面的直接视频命令。

## 跌倒报警证据

帧序列模式、视频文件模式和 ISP 摄像头模式都包含事件级跌倒报警证据功能。

程序会在内存中保留最近约 10 秒已绘制结果框的帧。在 ISP 摄像头模式下，帧缓存与 AI 推理解耦，默认约为 8 FPS，因此证据片段会比约 3 FPS 的 AI 循环更平滑。由于板端推理耗时可能不均匀，跌倒证据采用时间投票规则，而不是要求模型连续逐帧命中：当模型在 6 秒窗口内至少 3 次输出跌倒框时，程序会排队执行后台保存任务，并创建事件目录：

```text
/sharefs/sdcard/evidence/YYYYMMDD_HHMMSS/
|-- snapshot.jpg
`-- frames/
    |-- 000001.jpg
    `-- ...
```

如果 `/sharefs/sdcard/evidence` 不可写，程序会依次尝试 `/sdcard/evidence`，最后回退到当前工作目录下的 `./evidence`。

快照和缓存帧都会绘制检测到的跌倒框。帧序列包含确认跌倒前缓存的画面，最长约 10 秒。证据写入运行在分离的后台线程中，因此较慢的 SD 卡写入不会阻塞实时检测循环。默认使用 JPEG 帧序列，是因为某些 K230 OpenCV 部署中 `cv::VideoWriter::open` 可能卡住。缓存前会把证据 JPG 缩放到宽度不超过 640 像素，以降低板端 CPU、内存和 SD 卡写入压力。

AVI 输出是可选能力。只有显式开启并创建成功时才会生成 `event.avi`；默认情况或 AVI 失败时都会生成 `frames/`：

```text
<selected evidence root>/YYYYMMDD_HHMMSS/
|-- snapshot.jpg
|-- event.avi              # 仅在 FDD_EVIDENCE_AVI=1 且创建成功时生成，可选
`-- frames/                # 默认输出，AVI 失败时也使用该目录
    |-- 000001.jpg
    `-- ...
```

每次独立跌倒事件只保存一次。持续跌倒不会重复保存；只有跌倒检测消失约 3 秒后，新的投票窗口才可以再次触发保存。

需要时可以使用 `FDD_EVIDENCE_ROOT` 覆盖默认 SD 卡目录。在 RT-Smart `msh` 中，请先设置环境变量再运行程序：

```sh
setenv FDD_EVIDENCE_ROOT /sharefs/sdcard/evidence
./falldown_detect.elf best_fp16.kmodel None 0
```

只有在需要尝试板端创建 `event.avi` 时才设置 `FDD_EVIDENCE_AVI=1`。默认保持未设置，使用更稳妥的 JPG 帧序列。

如果希望调整摄像头模式下证据片段的平滑程度，可以在运行前设置 `FDD_ISP_EVIDENCE_FPS`。例如，在 RT-Smart `msh` 中：

```sh
setenv FDD_ISP_EVIDENCE_FPS 10
./falldown_detect.elf best_fp16.kmodel None 0
```

更高的数值会生成更平滑的帧序列，但也会增加 CPU、内存和 SD 卡压力。如果实时画面变得不稳定，可以尝试 `6` 或 `8`。

当板子需要在不连接显示屏的情况下运行时，可以设置 `FDD_DISPLAY`。默认值是 `auto`：程序会先尝试正常 VO/OSD 显示路径，如果显示初始化失败，则继续以无屏方式运行。通信测试时可使用 `off` 完全跳过显示初始化；如果希望强制要求显示可用，可使用 `on`，显示缺失时程序会快速失败并退出。

```sh
./falldown_split_launcher.elf best_fp16.kmodel 0 off
```

需要强调的是，`FDD_DISPLAY=off` 主要用于当前硬件无法同时稳定连接显示屏和网络时的通信链路测试，并不是作品的核心功能卖点。连接显示屏时用于验证本地预览和 OSD，拔掉显示屏联网时用于验证大小核通知、PC HTTP 服务和手机端事件读取。

测试证据状态时，可以用 `setenv` 设置 `FDD_EVIDENCE_DEBUG=1`。程序会每秒打印一行证据状态。`fall_box=1` 表示模型当前输出了跌倒框，`vote_hits` 表示当前 6 秒投票窗口内的跌倒命中次数。

## ONNX 测试工具

Windows 侧 ONNX 测试流程位于 `tools/onnx_test/`。

```powershell
cd tools\onnx_test
py -m pip install -r requirements.txt
py eval_onnx_windows.py --model ..\..\models\best_fp16.onnx --images calib_images --out outputs
```

## 主机侧证据同步

板端将跌倒证据写入 `/sharefs/sdcard/evidence` 后，Ubuntu 主机可以自动拉取新的事件目录并生成 MP4 片段：

```sh
cd ~/k230_sdk/src/big/mpp/userapps/sample/falldown_ai_inference_0606/data
python3 auto_evidence_service.py
```

默认情况下，辅助工具只监控最新的远端事件目录。这样可以避免旧的板端目录，例如 `19700101_000325`，阻塞当前流程。正常监控时，如果板端已经存在旧事件目录，建议使用 `--ignore-existing` 启动，使工具只处理启动之后新创建的事件：

```sh
python3 auto_evidence_service.py --ignore-existing --verbose
```

调试时如果只想拉取某一个已知事件，可以运行：

```sh
python3 auto_evidence_service.py --event 19700101_001429 --resync --verbose --once
```

如果确实需要回填板端所有事件目录，可以添加 `--all`。如果本地证据目录已经存在但希望强制重新从板端拉取，可以添加 `--resync`：

```sh
python3 auto_evidence_service.py --all --resync --verbose
```

默认设置匹配当前环境：远端根目录为 `/sharefs/sdcard/evidence`，本地根目录为 `/home/ubuntu/k230_sdk/src/big/mpp/userapps/sample/falldown_ai_inference_0606/data/evidence`，MP4 输出帧率为 `8` FPS。辅助工具会通过 adb 扫描，等待 `frames/` 数量稳定，拉取选中的事件目录，然后生成 `event.mp4` 和本地 `event.json`。
