package com.example.re

import android.graphics.Color
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import java.net.URI
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var statusText: TextView
    private lateinit var lastUpdatedText: TextView
    private lateinit var emptyText: TextView
    private lateinit var currentServerText: TextView
    private lateinit var serverAddressInput: EditText
    private lateinit var eventsContainer: LinearLayout
    private lateinit var refreshButton: Button
    private lateinit var applyServerButton: Button

    private lateinit var fallEventService: FallEventService
    private val refreshExecutor = Executors.newSingleThreadExecutor()
    private val events = mutableListOf<FallEvent>()

    private var watchHandle: WatchHandle? = null
    private var isBusy = false
    private var alarmDialog: AlertDialog? = null
    private var currentServerBaseUrl = AppConfig.API_BASE_URL
    private var serviceGeneration = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { view, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            view.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        bindViews()
        currentServerBaseUrl = loadServerBaseUrl()
        fallEventService = createFallEventService(currentServerBaseUrl)
        setupServerAddressControls()
        refreshButton.setOnClickListener { refreshEvents() }
        refreshEvents()
        startWatchingEvents()
    }

    override fun onDestroy() {
        stopWatchingEvents()
        alarmDialog?.dismiss()
        refreshExecutor.shutdownNow()
        super.onDestroy()
    }

    private fun bindViews() {
        statusText = findViewById(R.id.statusText)
        lastUpdatedText = findViewById(R.id.lastUpdatedText)
        emptyText = findViewById(R.id.emptyText)
        currentServerText = findViewById(R.id.currentServerText)
        serverAddressInput = findViewById(R.id.serverAddressInput)
        eventsContainer = findViewById(R.id.eventsContainer)
        refreshButton = findViewById(R.id.refreshButton)
        applyServerButton = findViewById(R.id.applyServerButton)
    }

    private fun setupServerAddressControls() {
        if (!AppConfig.USE_HTTP_SERVICE) {
            currentServerText.text = "当前模式：Mock"
            serverAddressInput.isEnabled = false
            applyServerButton.isEnabled = false
            return
        }

        serverAddressInput.setText(currentServerBaseUrl)
        updateCurrentServerText()
        applyServerButton.setOnClickListener { applyServerAddress() }
    }

    private fun applyServerAddress() {
        if (!AppConfig.USE_HTTP_SERVICE) {
            return
        }
        if (isBusy) {
            statusText.text = "请等待当前刷新完成后再切换地址"
            return
        }

        val rawValue = serverAddressInput.text?.toString().orEmpty()
        if (rawValue.isBlank()) {
            statusText.text = "请输入服务器地址"
            return
        }

        val nextBaseUrl = normalizeServerBaseUrl(rawValue)
        saveServerBaseUrl(nextBaseUrl)
        serverAddressInput.setText(nextBaseUrl)
        reconnectToServer(nextBaseUrl)
    }

    private fun reconnectToServer(baseUrl: String) {
        stopWatchingEvents()
        alarmDialog?.dismiss()
        events.clear()
        renderEvents()
        currentServerBaseUrl = baseUrl
        fallEventService = createFallEventService(baseUrl)
        serviceGeneration += 1
        updateCurrentServerText()
        statusText.text = "正在连接 $baseUrl"
        lastUpdatedText.text = "地址已更新，准备重新拉取"
        refreshEvents()
        startWatchingEvents()
    }

    private fun createFallEventService(baseUrl: String): FallEventService {
        return if (AppConfig.USE_HTTP_SERVICE) {
            HttpFallEventService(baseUrl)
        } else {
            MockFallEventService()
        }
    }

    private fun stopWatchingEvents() {
        watchHandle?.stop()
        watchHandle = null
    }

    private fun updateCurrentServerText() {
        currentServerText.text = "当前服务：$currentServerBaseUrl"
    }

    private fun loadServerBaseUrl(): String {
        val preferences = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
        val stored = preferences.getString(KEY_SERVER_BASE_URL, AppConfig.API_BASE_URL).orEmpty()
        return normalizeServerBaseUrl(stored.ifBlank { AppConfig.API_BASE_URL })
    }

    private fun saveServerBaseUrl(baseUrl: String) {
        getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
            .edit()
            .putString(KEY_SERVER_BASE_URL, baseUrl)
            .apply()
    }

    private fun normalizeServerBaseUrl(value: String): String {
        val trimmed = value.trim().trimEnd('/')
        if (trimmed.isBlank()) {
            return AppConfig.API_BASE_URL
        }

        val withScheme = if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
            trimmed
        } else {
            "http://$trimmed"
        }

        return appendDefaultPortIfMissing(withScheme).trimEnd('/')
    }

    private fun appendDefaultPortIfMissing(value: String): String {
        return try {
            val uri = URI(value)
            if (uri.host.isNullOrBlank() || uri.port != -1) {
                return value
            }

            val defaultUri = URI(AppConfig.API_BASE_URL)
            if (defaultUri.port == -1) {
                value
            } else {
                URI(
                    uri.scheme,
                    uri.userInfo,
                    uri.host,
                    defaultUri.port,
                    uri.path,
                    uri.query,
                    uri.fragment
                ).toString()
            }
        } catch (_: Exception) {
            value
        }
    }

    private fun refreshEvents() {
        if (isBusy) {
            return
        }

        isBusy = true
        refreshButton.isEnabled = false
        applyServerButton.isEnabled = false
        statusText.text = "正在刷新事件..."
        val activeService = fallEventService
        val generation = serviceGeneration

        refreshExecutor.execute {
            try {
                val latest = activeService.getRecentEvents().sortedByDescending { it.sequence }
                runOnUiThread {
                    if (generation != serviceGeneration) {
                        return@runOnUiThread
                    }
                    events.clear()
                    events.addAll(latest)
                    renderEvents()
                    lastUpdatedText.text = "最近刷新：${formatClockTime()}"
                    statusText.text = if (events.isEmpty()) "暂无事件" else "已加载 ${events.size} 条事件"
                    isBusy = false
                    refreshButton.isEnabled = true
                    applyServerButton.isEnabled = AppConfig.USE_HTTP_SERVICE
                }
            } catch (error: Throwable) {
                runOnUiThread {
                    if (generation != serviceGeneration) {
                        return@runOnUiThread
                    }
                    statusText.text = "刷新失败：${error.localizedMessage ?: "未知错误"}"
                    isBusy = false
                    refreshButton.isEnabled = true
                    applyServerButton.isEnabled = AppConfig.USE_HTTP_SERVICE
                }
            }
        }
    }

    private fun startWatchingEvents() {
        if (watchHandle != null) {
            return
        }

        val generation = serviceGeneration
        watchHandle = fallEventService.startWatching(
            pollIntervalMillis = 20_000L,
            onEvent = { item ->
                runOnUiThread {
                    if (generation != serviceGeneration) {
                        return@runOnUiThread
                    }
                    events.add(0, item)
                    renderEvents()
                    statusText.text = "收到新事件：${item.displayTime}"
                    lastUpdatedText.text = "最近推送：${formatClockTime()}"
                    showAlarm(item)
                }
            },
            onError = { error ->
                runOnUiThread {
                    if (generation != serviceGeneration) {
                        return@runOnUiThread
                    }
                    statusText.text = "监听中断：${error.localizedMessage ?: "未知错误"}"
                }
            }
        )
    }

    private fun renderEvents() {
        eventsContainer.removeAllViews()
        emptyText.visibility = if (events.isEmpty()) View.VISIBLE else View.GONE

        val inflater = LayoutInflater.from(this)
        events.forEach { item ->
            val itemView = inflater.inflate(R.layout.item_fall_event, eventsContainer, false)
            itemView.findViewById<TextView>(R.id.severityText).apply {
                text = item.severityText
                setTextColor(Color.parseColor(item.severityColor))
            }
            itemView.findViewById<TextView>(R.id.frameCountText).text = "${item.frameCount} 帧"
            itemView.findViewById<TextView>(R.id.messageText).text = item.message
            itemView.findViewById<TextView>(R.id.locationText).text = item.location
            itemView.findViewById<TextView>(R.id.displayTimeText).text = item.displayTime
            itemView.findViewById<TextView>(R.id.snapshotPathText).text = item.snapshotPath
            itemView.setOnClickListener { showEventDetail(item) }
            eventsContainer.addView(itemView)
        }
    }

    private fun showEventDetail(item: FallEvent) {
        AlertDialog.Builder(this)
            .setTitle(item.severityText)
            .setMessage(item.detailText)
            .setPositiveButton(R.string.fall_detail_close, null)
            .show()
    }

    private fun showAlarm(item: FallEvent) {
        alarmDialog?.dismiss()
        alarmDialog = AlertDialog.Builder(this)
            .setTitle(R.string.fall_alert_title)
            .setMessage(item.detailText)
            .setPositiveButton(R.string.fall_alert_confirm, null)
            .show()
    }

    private fun formatClockTime(): String {
        return SimpleDateFormat("HH:mm:ss", Locale.CHINA).format(Date())
    }

    private companion object {
        private const val PREFS_NAME = "fall_guard_prefs"
        private const val KEY_SERVER_BASE_URL = "server_base_url"
    }
}
