package com.example.re

interface FallEventService {
    fun getRecentEvents(): List<FallEvent>

    fun startWatching(
        pollIntervalMillis: Long,
        onEvent: (FallEvent) -> Unit,
        onError: (Throwable) -> Unit
    ): WatchHandle
}

interface WatchHandle {
    fun stop()
}