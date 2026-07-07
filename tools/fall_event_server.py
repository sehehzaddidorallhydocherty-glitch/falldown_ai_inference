#!/usr/bin/env python3
import argparse
import json
import mimetypes
from pathlib import Path
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
SIGNAL_AUDIO_PATH = SCRIPT_DIR / "signal.mp3"


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FallGuard Alarm Console</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      background: #f6f7f9;
      color: #17202a;
    }
    body {
      margin: 0;
      padding: 24px;
      background: #f6f7f9;
    }
    main {
      max-width: 980px;
      margin: 0 auto;
    }
    header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 20px;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 28px;
      line-height: 1.2;
    }
    .muted {
      color: #667085;
      font-size: 14px;
    }
    .panel {
      background: #ffffff;
      border: 1px solid #d0d5dd;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06);
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }
    button {
      border: 1px solid #98a2b3;
      border-radius: 6px;
      padding: 10px 14px;
      background: #ffffff;
      color: #17202a;
      cursor: pointer;
      font-size: 14px;
    }
    button.primary {
      border-color: #b42318;
      background: #d92d20;
      color: #ffffff;
    }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
    }
    .status {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 4px 10px;
      border-radius: 999px;
      background: #eef2ff;
      color: #3538cd;
      font-size: 13px;
    }
    .status.alarm {
      background: #fee4e2;
      color: #b42318;
    }
    .event-list {
      display: grid;
      gap: 10px;
    }
    .event {
      border: 1px solid #eaecf0;
      border-radius: 8px;
      padding: 12px;
      background: #fcfcfd;
    }
    .event-title {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
      font-weight: 700;
    }
    dl {
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 6px 12px;
      margin: 0;
      font-size: 14px;
    }
    dt {
      color: #667085;
    }
    dd {
      margin: 0;
      word-break: break-all;
    }
    code {
      background: #f2f4f7;
      border-radius: 4px;
      padding: 2px 4px;
    }
    @media (max-width: 640px) {
      body {
        padding: 14px;
      }
      header {
        display: block;
      }
      dl {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>FallGuard Alarm Console</h1>
        <div class="muted">保持此页面打开。小核上报新跌倒事件后，页面会循环播放报警音。</div>
      </div>
      <span id="alarmStatus" class="status">等待启用音频</span>
    </header>

    <section class="panel">
      <div class="controls">
        <button id="enableAudio" class="primary">启用报警音频</button>
        <button id="stopAudio" disabled>停止警报</button>
        <button id="refreshEvents">刷新事件</button>
      </div>
      <p class="muted">
        上报地址：<code id="postUrl"></code><br>
        浏览器限制自动播放，所以打开页面后请先点一次“启用报警音频”。
      </p>
      <audio id="alarmAudio" src="/signal.mp3" loop preload="auto"></audio>
    </section>

    <section class="panel">
      <h2>最近事件</h2>
      <div id="eventList" class="event-list"></div>
    </section>
  </main>

  <script>
    const alarmAudio = document.getElementById("alarmAudio");
    const alarmStatus = document.getElementById("alarmStatus");
    const enableAudio = document.getElementById("enableAudio");
    const stopAudio = document.getElementById("stopAudio");
    const refreshEvents = document.getElementById("refreshEvents");
    const eventList = document.getElementById("eventList");
    const postUrl = document.getElementById("postUrl");

    let audioEnabled = false;
    let lastSequence = 0;

    postUrl.textContent = `${location.origin}/api/fall-events`;

    function setStatus(text, alarm = false) {
      alarmStatus.textContent = text;
      alarmStatus.classList.toggle("alarm", alarm);
    }

    function renderEvents(events) {
      if (!events.length) {
        eventList.innerHTML = '<div class="muted">暂无事件</div>';
        return;
      }

      eventList.innerHTML = events.map((item) => `
        <article class="event">
          <div class="event-title">
            <span>${escapeHtml(item.eventId || "fall-event")}</span>
            <span>${escapeHtml(item.occurredAt || "")}</span>
          </div>
          <dl>
            <dt>消息</dt><dd>${escapeHtml(item.message || "")}</dd>
            <dt>设备</dt><dd>${escapeHtml(item.deviceName || "")}</dd>
            <dt>位置</dt><dd>${escapeHtml(item.location || "")}</dd>
            <dt>帧数</dt><dd>${Number(item.frameCount || 0)}</dd>
            <dt>证据目录</dt><dd>${escapeHtml(item.eventDir || "")}</dd>
            <dt>快照</dt><dd>${escapeHtml(item.snapshot || "")}</dd>
          </dl>
        </article>
      `).join("");
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    async function loadInitialEvents() {
      const response = await fetch("/api/fall-events", { cache: "no-store" });
      const events = await response.json();
      renderEvents(events);
      for (const item of events) {
        lastSequence = Math.max(lastSequence, Number(item.sequence || 0));
      }
      if (audioEnabled) {
        setStatus("已启用，等待新事件");
      }
    }

    async function pollEvents() {
      try {
        const response = await fetch(`/api/fall-events?afterSeq=${lastSequence}`, { cache: "no-store" });
        const events = await response.json();
        if (events.length) {
          for (const item of events) {
            lastSequence = Math.max(lastSequence, Number(item.sequence || 0));
          }
          await loadInitialEvents();
          triggerAlarm();
        }
      } catch (error) {
        setStatus(`连接中断：${error.message}`);
      }
    }

    function triggerAlarm() {
      if (!audioEnabled) {
        setStatus("收到新事件，请先启用音频", true);
        return;
      }
      alarmAudio.currentTime = 0;
      alarmAudio.play()
        .then(() => {
          stopAudio.disabled = false;
          setStatus("正在报警", true);
        })
        .catch((error) => {
          setStatus(`播放失败：${error.message}`, true);
        });
    }

    enableAudio.addEventListener("click", async () => {
      try {
        alarmAudio.muted = true;
        await alarmAudio.play();
        alarmAudio.pause();
        alarmAudio.currentTime = 0;
        alarmAudio.muted = false;
        audioEnabled = true;
        enableAudio.disabled = true;
        stopAudio.disabled = false;
        setStatus("已启用，等待新事件");
      } catch (error) {
        setStatus(`启用失败：${error.message}`, true);
      }
    });

    stopAudio.addEventListener("click", () => {
      alarmAudio.pause();
      alarmAudio.currentTime = 0;
      setStatus(audioEnabled ? "已停止，等待新事件" : "等待启用音频");
    });

    refreshEvents.addEventListener("click", loadInitialEvents);

    loadInitialEvents().catch((error) => setStatus(`加载失败：${error.message}`, true));
    setInterval(pollEvents, 2000);
  </script>
</body>
</html>
"""


class EventStore:
    def __init__(self, log_path=None):
        self.events = []
        self.next_sequence = 1000
        self.log_path = log_path

    def add(self, payload):
        self.next_sequence += 1
        event = self.normalize(payload, self.next_sequence)
        self.events.append(event)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as file:
                file.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    def list(self, after_seq=None):
        if after_seq is None:
            return list(reversed(self.events[-50:]))
        return [item for item in self.events if item["sequence"] > after_seq]

    @staticmethod
    def normalize(payload, sequence):
        now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        event_dir = payload.get("eventDir") or payload.get("event_dir") or ""
        frame_count = payload.get("frameCount", payload.get("frame_count", 0))
        event = {
            "sequence": sequence,
            "eventId": payload.get("eventId") or f"fall-{sequence}",
            "type": payload.get("type", "fall_detected"),
            "deviceName": payload.get("deviceName", "K230-RTSmart-01"),
            "location": payload.get("location", "Indoor"),
            "eventDir": event_dir,
            "snapshot": payload.get("snapshot", ""),
            "frameCount": int(frame_count or 0),
            "message": payload.get("message", "Fall Detected!"),
            "occurredAt": payload.get("occurredAt", now),
            "acknowledged": bool(payload.get("acknowledged", False)),
        }
        return event


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/" or parsed.path == "/index.html":
                self.send_html(200, INDEX_HTML)
                return

            if parsed.path == "/signal.mp3":
                self.send_file(SIGNAL_AUDIO_PATH)
                return

            if parsed.path != "/api/fall-events":
                self.send_json(404, {"ok": False, "error": "not found"})
                return

            query = parse_qs(parsed.query)
            after_seq = None
            if "afterSeq" in query:
                try:
                    after_seq = int(query["afterSeq"][0])
                except ValueError:
                    after_seq = None
            self.send_json(200, store.list(after_seq))

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/api/fall-events":
                self.send_json(404, {"ok": False, "error": "not found"})
                return

            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            try:
                payload = json.loads(body)
            except json.JSONDecodeError as exc:
                self.send_json(400, {"ok": False, "error": str(exc)})
                return

            event = store.add(payload)
            print(f"[fall-event-server] received {event['eventId']} {event['snapshot']}")
            self.send_json(200, {"ok": True, "sequence": event["sequence"], "eventId": event["eventId"]})

        def log_message(self, fmt, *args):
            return

        def send_json(self, code, payload):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def send_html(self, code, html):
            data = html.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_file(self, path):
            if not path.exists() or not path.is_file():
                self.send_json(404, {"ok": False, "error": "file not found"})
                return

            content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Fall event HTTP test server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--log", default="fall_events_server.log")
    args = parser.parse_args()

    store = EventStore(args.log)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print(f"[fall-event-server] listening on http://{args.host}:{args.port}")
    print("[fall-event-server] UI   /")
    print("[fall-event-server] POST /api/fall-events")
    print("[fall-event-server] GET  /api/fall-events")
    print("[fall-event-server] GET  /signal.mp3")
    if not SIGNAL_AUDIO_PATH.exists():
        print(f"[fall-event-server] warning: missing alarm audio: {SIGNAL_AUDIO_PATH}")
    server.serve_forever()


if __name__ == "__main__":
    main()
