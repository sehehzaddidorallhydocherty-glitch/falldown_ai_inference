# 证据同步服务

这是一个运行在主机侧的辅助工具，用于通过 adb 拉取 K230 板端生成的跌倒证据，并自动生成 MP4 片段。

默认路径与当前部署环境保持一致：

```text
board:  /sharefs/sdcard/evidence
host:   /home/ubuntu/k230_sdk/src/big/mpp/userapps/sample/falldown_ai_inference_0606/data/evidence
fps:    8
```

在 Ubuntu 主机上运行：

```sh
cd ~/k230_sdk/src/big/mpp/userapps/sample/falldown_ai_inference_0606/data
python3 auto_evidence_service.py
```

服务会扫描板端事件目录。默认情况下，它只关注最新的远端事件目录；等待 `snapshot.jpg` 存在且 `frames/` 数量稳定后，将该事件目录拉取到本地，然后执行：

```sh
ffmpeg -framerate 8 -i frames/%06d.jpg -c:v libx264 -pix_fmt yuv420p event.mp4
```

常用选项：

```sh
# 匹配板端 10 FPS 缓存设置。
python3 auto_evidence_service.py --fps 10

# 只扫描并处理一次，然后退出。
python3 auto_evidence_service.py --once

# 板端已经存在旧目录时的正常监控方式。
python3 auto_evidence_service.py --ignore-existing --verbose

# 调试时拉取一个已知板端事件。
python3 auto_evidence_service.py --event 19700101_001429 --resync --verbose --once

# 回填板端所有事件目录。
python3 auto_evidence_service.py --all --resync --verbose

# 强制重新生成本地 event.mp4 文件。
python3 auto_evidence_service.py --overwrite

# 本地目录过期时，强制重新拉取板端最新事件。
python3 auto_evidence_service.py --resync --verbose
```

每个已处理事件都会生成一个本地 `event.json`，其中记录拉取路径、帧数量、FPS、快照路径和生成的 MP4 路径。
