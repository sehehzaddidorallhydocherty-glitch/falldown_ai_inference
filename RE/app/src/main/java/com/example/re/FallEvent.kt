package com.example.re

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

enum class FallEventSeverity {
    INFO,
    WARNING,
    ALARM
}

data class FallEvent(
    val sequence: Long,
    val eventId: String,
    val occurredAtMillis: Long,
    val severity: FallEventSeverity = FallEventSeverity.ALARM,
    val deviceName: String = "K230 Fall Monitor",
    val location: String = "Indoor",
    val snapshotPath: String = "",
    val evidencePath: String = "",
    val frameCount: Int,
    val message: String = "Fall Detected!",
    val acknowledged: Boolean
) {
    val severityText: String
        get() = when (severity) {
            FallEventSeverity.ALARM -> "跌倒告警"
            FallEventSeverity.WARNING -> "疑似风险"
            FallEventSeverity.INFO -> "状态信息"
        }

    val severityColor: String
        get() = when (severity) {
            FallEventSeverity.ALARM -> "#DC2626"
            FallEventSeverity.WARNING -> "#D97706"
            FallEventSeverity.INFO -> "#0F766E"
        }

    val displayTime: String
        get() = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.CHINA).format(Date(occurredAtMillis))

    val detailText: String
        get() = "设备：$deviceName\n" +
            "位置：$location\n" +
            "时间：$displayTime\n" +
            "证据目录：$evidencePath\n" +
            "抓拍：$snapshotPath\n" +
            "帧数：$frameCount"
}
