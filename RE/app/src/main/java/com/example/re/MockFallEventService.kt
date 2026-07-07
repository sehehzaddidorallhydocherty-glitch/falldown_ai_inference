package com.example.re

import java.util.concurrent.Executors
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

class MockFallEventService : FallEventService {
    private val sequence = AtomicLong(1003)

    override fun getRecentEvents(): List<FallEvent> {
        val now = System.currentTimeMillis()
        return listOf(
            createEvent(1003, now - 2 * MINUTE, "客厅", 29, acknowledged = false),
            createEvent(1002, now - 18 * MINUTE, "卧室", 31, acknowledged = true),
            createEvent(1001, now - 64 * MINUTE, "走廊", 26, acknowledged = true)
        )
    }

    override fun startWatching(
        pollIntervalMillis: Long,
        onEvent: (FallEvent) -> Unit,
        onError: (Throwable) -> Unit
    ): WatchHandle {
        val executor = Executors.newSingleThreadScheduledExecutor()
        val future: ScheduledFuture<*> = executor.scheduleAtFixedRate(
            {
                try {
                    val next = sequence.incrementAndGet()
                    onEvent(createEvent(next, System.currentTimeMillis(), "客厅", 28 + (next % 4).toInt(), acknowledged = false))
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

    private fun createEvent(
        sequence: Long,
        occurredAtMillis: Long,
        location: String,
        frameCount: Int,
        acknowledged: Boolean
    ): FallEvent {
        val eventDir = "/sharefs/sdcard/evidence/mock_$sequence"
        return FallEvent(
            sequence = sequence,
            eventId = "fall-$sequence",
            occurredAtMillis = occurredAtMillis,
            severity = FallEventSeverity.ALARM,
            deviceName = "K230-RTSmart-01",
            location = location,
            evidencePath = eventDir,
            snapshotPath = "$eventDir/snapshot.jpg",
            frameCount = frameCount,
            message = "Fall Detected!",
            acknowledged = acknowledged
        )
    }

    private companion object {
        private const val MINUTE = 60_000L
    }
}
