package com.example.re

import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

class HttpFallEventService(private val apiBaseUrl: String) : FallEventService {
    private val lastSequence = AtomicLong(0)

    override fun getRecentEvents(): List<FallEvent> {
        val events = fetchEvents(null).sortedByDescending { it.sequence }
        events.maxOfOrNull { it.sequence }?.let { lastSequence.updateAndGet { old -> maxOf(old, it) } }
        return events
    }

    override fun startWatching(
        pollIntervalMillis: Long,
        onEvent: (FallEvent) -> Unit,
        onError: (Throwable) -> Unit
    ): WatchHandle {
        val executor = Executors.newSingleThreadScheduledExecutor()
        val future: ScheduledFuture<*> = executor.scheduleWithFixedDelay(
            {
                try {
                    val events = fetchEvents(lastSequence.get()).sortedBy { it.sequence }
                    events.forEach { event ->
                        lastSequence.updateAndGet { old -> maxOf(old, event.sequence) }
                        onEvent(event)
                    }
                } catch (error: Throwable) {
                    onError(error)
                }
            },
            pollIntervalMillis,
            pollIntervalMillis,
            TimeUnit.MILLISECONDS
        )

        return object : WatchHandle {
            override fun stop() {
                future.cancel(true)
                executor.shutdownNow()
            }
        }
    }

    private fun fetchEvents(afterSeq: Long?): List<FallEvent> {
        val suffix = if (afterSeq == null) {
            "/api/fall-events"
        } else {
            "/api/fall-events?afterSeq=$afterSeq"
        }
        val connection = URL(apiBaseUrl.trimEnd('/') + suffix).openConnection() as HttpURLConnection
        connection.requestMethod = "GET"
        connection.connectTimeout = 3000
        connection.readTimeout = 3000

        val code = connection.responseCode
        if (code !in 200..299) {
            throw IllegalStateException("HTTP $code")
        }

        val body = BufferedReader(InputStreamReader(connection.inputStream)).use { it.readText() }
        val array = JSONArray(body)
        return buildList {
            for (index in 0 until array.length()) {
                add(parseEvent(array.getJSONObject(index)))
            }
        }
    }

    private fun parseEvent(json: JSONObject): FallEvent {
        val sequence = json.optLong("sequence", System.currentTimeMillis())
        val occurredAtMillis = parseOccurredAt(json.optString("occurredAt", ""))
        val type = json.optString("type", "fall_detected")
        return FallEvent(
            sequence = sequence,
            eventId = json.optString("eventId", "fall-$sequence"),
            occurredAtMillis = occurredAtMillis,
            severity = if (type == "fall_detected") FallEventSeverity.ALARM else FallEventSeverity.INFO,
            deviceName = json.optString("deviceName", "K230 Fall Monitor"),
            location = json.optString("location", "Indoor"),
            evidencePath = json.optString("eventDir", ""),
            snapshotPath = json.optString("snapshot", ""),
            frameCount = json.optInt("frameCount", 0),
            message = json.optString("message", "Fall Detected!"),
            acknowledged = json.optBoolean("acknowledged", false)
        )
    }

    private fun parseOccurredAt(value: String): Long {
        if (value.isBlank()) {
            return System.currentTimeMillis()
        }

        val formats = listOf(
            "yyyy-MM-dd'T'HH:mm:ssXXX",
            "yyyy-MM-dd'T'HH:mm:ssZ",
            "yyyy-MM-dd HH:mm:ss"
        )
        for (pattern in formats) {
            try {
                val parsed = SimpleDateFormat(pattern, Locale.US).parse(value)
                if (parsed != null) {
                    return parsed.time
                }
            } catch (_: Exception) {
            }
        }
        return System.currentTimeMillis()
    }
}
