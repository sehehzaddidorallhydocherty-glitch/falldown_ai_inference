# Linux 小核跌倒通知客户端

`falldown_notify_client` 运行在 K230 Linux 小核侧。它通过官方大小核通信样例接收 RT-Smart 大核发送的跌倒事件，然后执行两件事：

- 将每条事件 JSON 追加写入本地 JSONL 日志文件。
- 可选地把事件 JSON 推送到 HTTP webhook。

## 编译

```sh
cd ~/k230_sdk/src/big/mpp/userapps/sample/falldown_ai_inference_0613/linux_notify
make clean
make K230_SDK_ROOT=/home/ubuntu/k230_sdk
```

推送到板端：

```sh
adb push ./falldown_notify_client /sharefs/
adb shell chmod +x /sharefs/falldown_notify_client
adb shell sync
```

## 不启用 HTTP 推送运行

这种方式保持之前已经验证过的行为，只增加本地事件日志记录。

```sh
cd /sharefs
./falldown_notify_client
```

默认日志路径：

```text
/sharefs/sdcard/fall_events.log
```

日志中每一行都是一条 JSON：

```json
{"received_at":"2026-07-02T12:00:00+0800","event":{"type":"fall_detected","event_dir":"/sharefs/sdcard/evidence/xxx","snapshot":"/sharefs/sdcard/evidence/xxx/snapshot.jpg","frame_count":29}}
```

验证日志：

```sh
tail -n 5 /sharefs/sdcard/fall_events.log
```

## 启用 HTTP 推送运行

先在电脑上运行测试服务：

```sh
python tools/fall_event_server.py --host 0.0.0.0 --port 8080
```

然后在小核侧使用电脑局域网 IP 启动客户端：

```sh
cd /sharefs
./falldown_notify_client /sharefs/sdcard/fall_events.log http://192.168.1.100:8080/api/fall-events
```

也可以使用环境变量：

```sh
export FALL_NOTIFY_LOG=/sharefs/sdcard/fall_events.log
export FALL_NOTIFY_URL=http://192.168.1.100:8080/api/fall-events
./falldown_notify_client
```

第一版只支持普通 HTTP：

```text
http://host[:port]/path
```

暂未加入 HTTPS，目的是避免在板端客户端中引入额外 TLS 库，降低交叉编译和部署复杂度。

## 关闭本地日志

日志路径传入 `-` 即可关闭本地日志写入：

```sh
./falldown_notify_client - http://192.168.1.100:8080/api/fall-events
```

## 启停顺序

```text
Start: RT-Smart big core first, Linux little-core client second
Stop: Linux little-core client first, RT-Smart big core second
```

按上述顺序启停可以减少关闭时出现的 IPC 断开提示。
