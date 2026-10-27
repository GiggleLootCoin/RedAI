package com.giggleloot.redai

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import dev.ffmpegkit.llama.Llama
import dev.ffmpegkit.llama.LlamaConfig
import dev.ffmpegkit.llama.LlamaModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

data class Message(val role: String, val text: String)

class MainActivity : ComponentActivity() {
    private var model: LlamaModel? = null
    private var modelFile: File? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        render()
    }

    private fun render(status: String? = null) {
        setContent {
            RedAIApp(
                onPickModel = { launchModelPicker() },
                onSend = { text, callback -> generate(text, callback) },
                isModelLoaded = model?.isLoaded == true,
                modelName = modelFile?.name,
                status = status
            )
        }
    }

    private fun launchModelPicker() {
        startActivityForResult(
            Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                type = "*/*"
                addCategory(Intent.CATEGORY_OPENABLE)
            },
            9001
        )
    }

    @Deprecated("Compatibility with Android 12")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != 9001 || resultCode != RESULT_OK) return
        val uri = data?.data ?: return
        render("Loading RedAI.gguf locally…")

        lifecycleScope.launch {
            try {
                val copied = copyModel(uri) ?: throw IllegalStateException("Could not read the selected file")
                model?.let { Llama.releaseModel(it) }
                model = withContext(Dispatchers.Default) {
                    Llama.loadModel(
                        modelPath = copied.absolutePath,
                        config = LlamaConfig(
                            contextSize = 2048,
                            threads = 4,
                            gpuLayers = 0,
                            temperature = 0.70f,
                            topP = 0.80f,
                            topK = 20
                        )
                    )
                }
                modelFile = copied
                render("RedAI loaded. Inference is running locally on this phone.")
            } catch (e: Exception) {
                model = null
                render("Could not load that GGUF: " + (e.message ?: "unknown error"))
            }
        }
    }

    private suspend fun copyModel(uri: Uri): File? = withContext(Dispatchers.IO) {
        runCatching {
            val dir = File(getExternalFilesDir("models"), "").apply { mkdirs() }
            val out = File(dir, "RedAI.gguf")
            contentResolver.openInputStream(uri)?.use { input ->
                out.outputStream().use { output -> input.copyTo(output, 1024 * 1024) }
            } ?: return@runCatching null
            out
        }.getOrNull()
    }

    private fun generate(userText: String, callback: (String) -> Unit) {
        val loaded = model ?: run {
            callback("Load RedAI.gguf first.")
            return
        }

        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.Default) {
                    Llama.complete(
                        model = loaded,
                        prompt = userText,
                        systemPrompt = "You are RedAI, created by Red (Little Red Big Smile). Answer directly, naturally, honestly and usefully. You run locally in RedUI. Never reveal hidden reasoning or response-generation planning.",
                        maxTokens = 512
                    )
                }
                callback(result.text.trim())
            } catch (e: Exception) {
                callback("RedAI runtime error: " + (e.message ?: "unknown error"))
            }
        }
    }

    override fun onDestroy() {
        model?.let { runCatching { Llama.releaseModel(it) } }
        model = null
        super.onDestroy()
    }
}

@Composable
fun RedAIApp(
    onPickModel: () -> Unit,
    onSend: (String, (String) -> Unit) -> Unit,
    isModelLoaded: Boolean,
    modelName: String?,
    status: String?
) {
    var input by remember { mutableStateOf("") }
    var messages by remember {
        mutableStateOf(listOf(Message("assistant", "RedAI is ready. Choose the RedAI.gguf model to start private, on-device chat.")))
    }
    var showSettings by remember { mutableStateOf(false) }
    var busy by remember { mutableStateOf(false) }

    MaterialTheme {
        Column(Modifier.fillMaxSize().background(Color(0xFFF7F7F8))) {
            Row(Modifier.fillMaxWidth().padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("RedAI", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                    Text(
                        if (isModelLoaded) "LOCAL • " + (modelName ?: "GGUF") else "OFFLINE-FIRST LOCAL AI",
                        style = MaterialTheme.typography.bodySmall
                    )
                }
                TextButton(onClick = { showSettings = !showSettings }) { Text("Settings") }
            }

            if (status != null) {
                Card(Modifier.fillMaxWidth().padding(horizontal = 16.dp), shape = RoundedCornerShape(14.dp)) {
                    Text(status, Modifier.padding(14.dp), style = MaterialTheme.typography.bodySmall)
                }
                Spacer(Modifier.height(8.dp))
            }

            if (!isModelLoaded) {
                Card(Modifier.fillMaxWidth().padding(horizontal = 16.dp), shape = RoundedCornerShape(18.dp)) {
                    Column(Modifier.padding(18.dp)) {
                        Text("RedAI GGUF", fontWeight = FontWeight.Bold)
                        Spacer(Modifier.height(6.dp))
                        Text("Load the RedAI Q4_K_M model. Inference stays on the phone; no paid API or cloud service is required.")
                        Spacer(Modifier.height(12.dp))
                        Button(onClick = onPickModel, modifier = Modifier.fillMaxWidth()) {
                            Text("Choose RedAI.gguf")
                        }
                    }
                }
            } else {
                Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    AssistChip(onClick = onPickModel, label = { Text("Change model") })
                    AssistChip(
                        onClick = { messages = listOf(Message("assistant", "New chat started.")) },
                        label = { Text("New chat") }
                    )
                }
            }

            if (showSettings) {
                Card(Modifier.fillMaxWidth().padding(16.dp), shape = RoundedCornerShape(18.dp)) {
                    Column(Modifier.padding(16.dp)) {
                        Text("Galaxy A12 preset", fontWeight = FontWeight.Bold)
                        Text("Qwen3-1.7B • Q4_K_M • 2048 context • 4 CPU threads")
                        Text("Temperature 0.70 • Top-p 0.80 • Top-k 20", style = MaterialTheme.typography.bodySmall)
                        Text("Qwen3 non-thinking mode is used for direct, cleaner mobile replies.", style = MaterialTheme.typography.bodySmall)
                    }
                }
            }

            LazyColumn(
                Modifier.weight(1f).fillMaxWidth().padding(horizontal = 16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
                contentPadding = PaddingValues(vertical = 16.dp)
            ) {
                items(messages) { message ->
                    Row(
                        Modifier.fillMaxWidth(),
                        horizontalArrangement = if (message.role == "user") Arrangement.End else Arrangement.Start
                    ) {
                        Surface(
                            shape = RoundedCornerShape(18.dp),
                            color = if (message.role == "user") Color(0xFFE5E5E7) else Color.White
                        ) {
                            Text(message.text, Modifier.padding(13.dp).widthIn(max = 320.dp))
                        }
                    }
                }
            }

            Row(Modifier.fillMaxWidth().padding(12.dp), verticalAlignment = Alignment.Bottom) {
                OutlinedTextField(
                    value = input,
                    onValueChange = { input = it },
                    modifier = Modifier.weight(1f),
                    placeholder = { Text("Message RedAI…") },
                    maxLines = 5,
                    enabled = !busy
                )
                Spacer(Modifier.width(8.dp))
                Button(
                    enabled = input.isNotBlank() && isModelLoaded && !busy,
                    onClick = {
                        val q = input.trim()
                        input = ""
                        messages = messages + Message("user", q)
                        busy = true
                        onSend(q) { answer ->
                            messages = messages + Message("assistant", answer)
                            busy = false
                        }
                    }
                ) {
                    Text(if (busy) "…" else "Send")
                }
            }
        }
    }
}
