package com.adamxit.glowupai

import android.app.Activity
import android.graphics.Typeface
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.InputType
import android.util.TypedValue
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.OutputStreamWriter
import java.io.Writer
import java.nio.charset.StandardCharsets
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : Activity() {

    private companion object {
        const val APP_NAME = "Glow Up Ai"
        const val DEVELOPER = "Adam xit"
        const val ABOUT = "Glow Up Ai - Developed by Adam xit - Fully offline"
        const val ANSWER_NAME = "Glow Up Ai"
        const val ANSWER_DEVELOPER = "Adam xit"
        const val ANSWER_IDENTITY = "I am Glow Up Ai, developed by Adam xit."
        const val MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
        const val MAX_TURNS = 3
        const val CTX_SIZE = 1024
        const val MAX_TOKENS = 120
        const val TIMEOUT_MS = 300_000L

        val NAME_PATTERNS = listOf(
            Regex("what(?:'s| is) your name", RegexOption.IGNORE_CASE),
            Regex("what do (?:they|people) call you", RegexOption.IGNORE_CASE),
            Regex("ما اسمك", RegexOption.IGNORE_CASE),
            Regex("ما هو اسمك", RegexOption.IGNORE_CASE),
            Regex("إيش اسمك", RegexOption.IGNORE_CASE),
            Regex("شنو اسمك", RegexOption.IGNORE_CASE)
        )
        val DEV_PATTERNS = listOf(
            Regex("who (?:developed|made|created|built) you", RegexOption.IGNORE_CASE),
            Regex("who is your (?:developer|creator|maker)", RegexOption.IGNORE_CASE),
            Regex("من صنعك", RegexOption.IGNORE_CASE),
            Regex("من طورك", RegexOption.IGNORE_CASE),
            Regex("من أنشأك", RegexOption.IGNORE_CASE),
            Regex("مين طورك", RegexOption.IGNORE_CASE)
        )
        val IDENTITY_PATTERNS = listOf(
            Regex("who are you", RegexOption.IGNORE_CASE),
            Regex("من أنت", RegexOption.IGNORE_CASE),
            Regex("مين انت", RegexOption.IGNORE_CASE)
        )
    }

    private val io: ExecutorService = Executors.newSingleThreadExecutor()
    private val ui = Handler(Looper.getMainLooper())
    private val history = mutableListOf<Entry>()
    private val busy = AtomicBoolean(false)

    private lateinit var transcript: TextView
    private lateinit var status: TextView
    private lateinit var input: EditText
    private lateinit var sendButton: Button
    private lateinit var scrollView: ScrollView
    private var systemPrompt = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        title = APP_NAME
        buildUi()
        transcript.text = ABOUT + "\n\n"
        io.execute { prepare() }
    }

    override fun onDestroy() {
        io.execute { LlamaEngine.unload() }
        io.shutdown()
        super.onDestroy()
    }

    private fun buildUi() {
        val pad = dp(16)

        transcript = TextView(this).apply {
            typeface = Typeface.MONOSPACE
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f)
            isTextSelectable = true
            textDirection = View.TEXT_DIRECTION_FIRST_STRONG
            textAlignment = View.TEXT_ALIGNMENT_VIEW_START
            setPadding(pad, pad, pad, pad)
        }
        scrollView = ScrollView(this).apply {
            addView(transcript, ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
        }

        status = TextView(this).apply {
            text = "Preparing offline model..."
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
            setPadding(pad, dp(4), pad, dp(4))
        }

        input = EditText(this).apply {
            hint = "Type a message..."
            isSingleLine = true
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
            textDirection = View.TEXT_DIRECTION_FIRST_STRONG
            isEnabled = false
        }
        sendButton = Button(this).apply {
            text = "Send"
            isEnabled = false
            setOnClickListener { onSend() }
        }
        val clearButton = Button(this).apply {
            text = "Clear"
            setOnClickListener { clearHistory() }
        }
        val aboutButton = Button(this).apply {
            text = "About"
            setOnClickListener { append(APP_NAME, ABOUT) }
        }

        val inputRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(input, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            addView(sendButton, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT))
            addView(clearButton, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT))
            addView(aboutButton, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        }

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutDirection = View.LAYOUT_DIRECTION_LOCALE
            addView(scrollView, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
            addView(status, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
            addView(inputRow, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
            setPadding(0, pad, 0, pad)
        }
        setContentView(root)
    }

    private fun prepare() {
        try {
            ui.post { status.text = "Copying bundled model to private storage..." }
            val model = ensureModelFile()
            systemPrompt = readAsset("system_prompt.txt")
            loadHistory()
            ui.post { status.text = "Loading model..." }
            val threads = minOf(4, maxOf(1, Runtime.getRuntime().availableProcessors()))
            if (!LlamaEngine.load(model.absolutePath, CTX_SIZE, threads)) {
                throw IOException("model load failed")
            }
            ui.post {
                status.text = "Ready - offline"
                input.isEnabled = true
                sendButton.isEnabled = true
            }
        } catch (exc: Exception) {
            val message = exc.message ?: exc.toString()
            ui.post {
                append("System", "Error: $message")
                status.text = "Startup failed"
            }
        }
    }

    private fun onSend() {
        val text = input.text.toString().trim()
        if (text.isEmpty()) return
        input.setText("")

        when (text.lowercase(Locale.ROOT)) {
            "/clear" -> {
                clearHistory()
                return
            }
            "/about" -> {
                append(APP_NAME, ABOUT)
                return
            }
            "/exit", "/quit", "exit", "quit" -> {
                append(APP_NAME, "Goodbye.")
                finish()
                return
            }
        }

        append("You", text)
        val fixed = identityAnswer(text)
        if (fixed != null) {
            append(APP_NAME, fixed)
            record(text, fixed)
            return
        }
        if (!LlamaEngine.isLoaded()) {
            append(APP_NAME, "Error: model is not ready.")
            return
        }
        if (!busy.compareAndSet(false, true)) return

        val prompt = buildPrompt(text)
        sendButton.isEnabled = false
        status.text = "Thinking..."
        val cancelled = AtomicBoolean(false)
        val timeout = Runnable {
            if (!cancelled.get()) {
                cancelled.set(true)
                append(APP_NAME, "Error: the model timed out after 300 seconds.")
                finishTurn()
            }
        }
        ui.postDelayed(timeout, TIMEOUT_MS)

        io.execute {
            try {
                val collected = StringBuilder()
                val answer = LlamaEngine.generate(
                    prompt,
                    MAX_TOKENS,
                    0.7f,
                    1.05f,
                    LlamaEngine.TokenCallback { collected.append(it) }
                )
                ui.post {
                    ui.removeCallbacks(timeout)
                    if (!cancelled.get()) {
                        val value = answer.trim()
                        if (value.isEmpty() || value.startsWith("Error:")) {
                            append(APP_NAME, "Error: empty response.")
                        } else {
                            append(APP_NAME, value)
                            record(text, value)
                        }
                    }
                    finishTurn()
                }
            } catch (exc: Exception) {
                ui.post {
                    ui.removeCallbacks(timeout)
                    if (!cancelled.get()) {
                        append(APP_NAME, "Error: " + (exc.message ?: exc.toString()))
                    }
                    finishTurn()
                }
            }
        }
    }

    private fun finishTurn() {
        busy.set(false)
        status.text = "Ready - offline"
        sendButton.isEnabled = true
    }

    private fun clearHistory() {
        try {
            FileOutputStream(historyFile(), false).close()
        } catch (exc: IOException) {
            append("System", "Error: could not clear history.")
            return
        }
        synchronized(history) { history.clear() }
        append(APP_NAME, "History cleared.")
    }

    private fun record(userText: String, answer: String) {
        synchronized(history) {
            history.add(Entry("user", userText))
            history.add(Entry("assistant", answer))
            saveTurn("user", userText)
            saveTurn("assistant", answer)
        }
    }

    private fun append(speaker: String, message: String) {
        ui.post {
            transcript.append("$speaker: $message\n\n")
            scrollView.post { scrollView.fullScroll(View.FOCUS_DOWN) }
        }
    }

    private fun identityAnswer(text: String): String? {
        if (matches(DEV_PATTERNS, text)) return ANSWER_DEVELOPER
        if (matches(NAME_PATTERNS, text)) return ANSWER_NAME
        if (matches(IDENTITY_PATTERNS, text)) return ANSWER_IDENTITY
        return null
    }

    private fun matches(patterns: List<Regex>, text: String): Boolean =
        patterns.any { it.containsMatchIn(text) }

    private fun buildPrompt(userText: String): String {
        val prompt = StringBuilder()
        prompt.append("<|im_start|>system\n")
            .append(systemPrompt.trim())
            .append("\n<|im_end|>\n")
        synchronized(history) {
            val start = maxOf(0, history.size - MAX_TURNS * 2)
            for (i in start until history.size) {
                val entry = history[i]
                prompt.append("<|im_start|>").append(entry.role).append("\n")
                    .append(entry.content).append("\n<|im_end|>\n")
            }
        }
        prompt.append("<|im_start|>user\n").append(userText).append("\n<|im_end|>\n")
        prompt.append("<|im_start|>assistant\n")
        return prompt.toString()
    }

    private fun loadHistory() {
        val file = historyFile()
        if (!file.exists()) return
        try {
            val content = String(file.readBytes(), StandardCharsets.UTF_8)
            val loaded = mutableListOf<Entry>()
            for (line in content.lineSequence()) {
                if (line.isBlank()) continue
                try {
                    val item = JSONObject(line)
                    val role = item.optString("role", "")
                    val body = item.optString("content", "")
                    if ((role == "user" || role == "assistant") && body.isNotEmpty()) {
                        loaded.add(Entry(role, body))
                    }
                } catch (_: Exception) {
                }
            }
            synchronized(history) {
                history.clear()
                history.addAll(loaded)
            }
        } catch (_: Exception) {
            append("System", "Warning: saved history could not be read.")
        }
    }

    private fun saveTurn(role: String, content: String) {
        try {
            val item = JSONObject()
            item.put("role", role)
            item.put("content", content)
            val writer: Writer = OutputStreamWriter(
                FileOutputStream(historyFile(), true), StandardCharsets.UTF_8)
            writer.use {
                it.write(item.toString())
                it.write('\n'.code)
            }
        } catch (_: Exception) {
            append("System", "Warning: history could not be saved.")
        }
    }

    private fun historyFile(): File = File(filesDir, "history.jsonl")

    private fun ensureModelFile(): File {
        val dir = File(filesDir, "models")
        if (!dir.exists() && !dir.mkdirs()) {
            throw IOException("Could not create model directory")
        }
        val target = File(dir, MODEL_FILE)
        val expected = assets.openFd(MODEL_FILE).use { it.length }
        if (target.exists() && target.length() == expected) return target

        val partial = File(dir, "$MODEL_FILE.part")
        assets.open(MODEL_FILE).use { input ->
            FileOutputStream(partial).use { output ->
                val buffer = ByteArray(64 * 1024)
                while (true) {
                    val read = input.read(buffer)
                    if (read < 0) break
                    output.write(buffer, 0, read)
                }
                output.flush()
            }
        }
        if (!partial.renameTo(target)) {
            throw IOException("Could not install model file")
        }
        return target
    }

    private fun readAsset(name: String): String {
        assets.open(name).use { input ->
            val output = ByteArrayOutputStream()
            val buffer = ByteArray(4096)
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                output.write(buffer, 0, read)
            }
            return String(output.toByteArray(), StandardCharsets.UTF_8).trim()
        }
    }

    private fun dp(value: Int): Int = TypedValue.applyDimension(
        TypedValue.COMPLEX_UNIT_DIP, value.toFloat(), resources.displayMetrics
    ).toInt()

    private data class Entry(val role: String, val content: String)
}
