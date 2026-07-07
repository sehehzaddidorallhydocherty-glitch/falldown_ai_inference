# K230 镜像打包辅助工具

本目录包含用于把跌倒检测运行文件预置到 K230 SDK 完整镜像中的辅助脚本。

该脚本设计为在 Ubuntu K230 SDK 环境中运行，项目目录应复制到：

```text
~/k230_sdk/src/big/mpp/userapps/sample/falldown_ai_inference_0613
```

## 安装内容

脚本会把以下文件放入 SDK Linux 小核 rootfs 的预置区域：

```text
/sharefs/falldown/
|-- falldown_ai_event.elf
|-- falldown_video_display.elf
|-- falldown_split_launcher.elf
|-- falldown_notify_client
|-- best_fp16.kmodel
|-- start_linux_notify.sh
|-- start_rtsmart_falldown.sh
`-- falldown_rtsmart_autostart.msh
```

脚本还会把兼容文件直接复制到 `/sharefs` 下，例如 `/sharefs/falldown_split_launcher.elf`。这里使用真实文件复制而不是符号链接，是因为 RT-Smart sharefs 对符号链接的支持可能随固件版本不同而变化。

## 运行

```sh
cd ~/k230_sdk/src/big/rt-smart
source smart-env.sh riscv64
cd ~/k230_sdk/src/big/mpp/userapps/sample/falldown_ai_inference_0613
chmod +x tools/image_packaging/install_to_k230_image.sh
tools/image_packaging/install_to_k230_image.sh \
  --sdk-root /home/ubuntu/k230_sdk \
  --output-config k230_canmv_dongshanpi_defconfig \
  --model /path/to/best_fp16.kmodel
```

如果 SDK 输出目录中只有一个 config，可以省略 `--output-config`。如果 `best_fp16.kmodel` 已经位于项目根目录或 `models/` 目录，也可以省略 `--model`。

## 构建最终 SDK 镜像

文件预置完成后，回到 SDK 根目录，运行当前板卡原本使用的镜像构建命令：

```sh
cd ~/k230_sdk
make
```

最终镜像通常位于：

```text
~/k230_sdk/output/<defconfig>/images/
```

## 自启动行为

Linux 小核自启动脚本安装为：

```text
/etc/init.d/S99falldown_notify
```

它会启动：

```sh
/sharefs/falldown/falldown_notify_client /sharefs/sdcard/fall_events.log
```

RT-Smart 自启动辅助脚本会预置到：

```text
/sharefs/falldown/falldown_rtsmart_autostart.msh
```

如果当前 SDK 已有明确的 RT-Smart 自动命令脚本，可以在其中调用该辅助脚本。如果没有，也可以手动启动大核运行程序：

```sh
cd /sharefs/falldown
./falldown_split_launcher.elf best_fp16.kmodel 0
```

## 烧录后验证

在 RT-Smart 侧：

```sh
cd /sharefs
ls
cd /sharefs/falldown
./falldown_split_launcher.elf best_fp16.kmodel 0
```

在 Linux 侧：

```sh
tail -n 5 /sharefs/sdcard/fall_events.log
```
