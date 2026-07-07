# FallGuard 原生 Android App

这是 K230 跌倒检测项目移动端使用的原生 Android Studio App。

## 当前模式

App 默认仍运行在 mock 模式：

```kotlin
object AppConfig {
    const val USE_HTTP_SERVICE = false
    const val API_BASE_URL = "http://192.168.1.100:8080"
}
```

在 Android Studio 中打开项目目录：

```text
RE/
```

可在 Android 手机或模拟器上运行。mock 模式不需要连接 K230 板子。

## 启用 HTTP 事件轮询

当已经有 HTTP API 服务接收 K230 小核推送的事件时，编辑：

```text
app/src/main/java/com/example/re/AppConfig.kt
```

设置：

```kotlin
const val USE_HTTP_SERVICE = true
const val API_BASE_URL = "http://your-server-ip:8080"
```

App 会调用：

```http
GET /api/fall-events
GET /api/fall-events?afterSeq=<lastSequence>
```

Android manifest 已经启用：

```xml
android.permission.INTERNET
android:usesCleartextTraffic="true"
```

因此局域网 HTTP 地址可以直接用于测试。

## 期望的事件 JSON

```json
[
  {
    "sequence": 1004,
    "eventId": "fall-1004",
    "type": "fall_detected",
    "deviceName": "K230-RTSmart-01",
    "location": "Indoor",
    "eventDir": "/sharefs/sdcard/evidence/19700101_030255",
    "snapshot": "/sharefs/sdcard/evidence/19700101_030255/snapshot.jpg",
    "frameCount": 29,
    "message": "Fall Detected!",
    "occurredAt": "2026-07-02T12:00:00+08:00",
    "acknowledged": false
  }
]
```

## K230 小核推送接口

小核客户端可以把每个事件推送到：

```http
POST /api/fall-events
Content-Type: application/json
```

请求体是 RT-Smart 大核生成的事件 JSON，例如：

```json
{
  "type": "fall_detected",
  "event_dir": "/sharefs/sdcard/evidence/19700101_030255",
  "snapshot": "/sharefs/sdcard/evidence/19700101_030255/snapshot.jpg",
  "frame_count": 29
}
```

后端服务可以把板端使用的 snake_case 字段转换为 Android App 使用的 camelCase 字段。

## PC 快速测试服务

局域网测试时，在电脑端项目根目录运行：

```sh
python tools/fall_event_server.py --host 0.0.0.0 --port 8080
```

然后把 `AppConfig.kt` 中的 `API_BASE_URL` 设置为电脑局域网 IP，例如：

```kotlin
const val API_BASE_URL = "http://192.168.1.100:8080"
```
