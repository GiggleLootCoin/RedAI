package com.giggleloot.redai

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
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

data class Message(val role: String, val text: String)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { RedAIApp() }
    }
}

@Composable
fun RedAIApp() {
    var input by remember { mutableStateOf("") }
    var modelLoaded by remember { mutableStateOf(false) }
    var messages by remember { mutableStateOf(listOf(Message("assistant", "RedAI is ready. Load a GGUF model to begin local chat."))) }
    var showSettings by remember { mutableStateOf(false) }
    var thinking by remember { mutableStateOf(false) }

    MaterialTheme {
        Column(Modifier.fillMaxSize().background(Color(0xFFF7F7F8))) {
            Row(Modifier.fillMaxWidth().padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("RedAI", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                    Text(if (modelLoaded) "Local model loaded" else "Offline-first local AI", style = MaterialTheme.typography.bodySmall)
                }
                TextButton(onClick = { showSettings = !showSettings }) { Text("Settings") }
            }

            if (!modelLoaded) {
                Card(Modifier.fillMaxWidth().padding(horizontal = 16.dp), shape = RoundedCornerShape(18.dp)) {
                    Column(Modifier.padding(18.dp)) {
                        Text("RedAI GGUF", fontWeight = FontWeight.Bold)
                        Spacer(Modifier.height(6.dp))
                        Text("Use the Q4_K_M RedAI model built for Galaxy A12-class hardware.")
                        Spacer(Modifier.height(12.dp))
                        Button(onClick = { modelLoaded = true }, modifier = Modifier.fillMaxWidth()) {
                            Text("Load RedAI.gguf")
                        }
                        Spacer(Modifier.height(6.dp))
                        Text("No paid API or cloud inference is required.", style = MaterialTheme.typography.bodySmall)
                    }
                }
            }

            if (showSettings) {
                Card(Modifier.fillMaxWidth().padding(16.dp), shape = RoundedCornerShape(18.dp)) {
                    Column(Modifier.padding(16.dp)) {
                        Text("Runtime", fontWeight = FontWeight.Bold)
                        Text("Qwen3-1.7B • Q4_K_M • 2048 context • 4 CPU threads")
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text("Direct mode", Modifier.weight(1f))
                            Switch(checked = !thinking, onCheckedChange = { thinking = !it })
                        }
                        Text("Thinking mode is off by default for faster, cleaner mobile replies.", style = MaterialTheme.typography.bodySmall)
                    }
                }
            }

            LazyColumn(Modifier.weight(1f).fillMaxWidth().padding(horizontal = 16.dp), verticalArrangement = Arrangement.spacedBy(10.dp), contentPadding = PaddingValues(vertical = 16.dp)) {
                items(messages) { m ->
                    val mine = m.role == "user"
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = if (mine) Arrangement.End else Arrangement.Start) {
                        Surface(shape = RoundedCornerShape(18.dp), color = if (mine) Color(0xFFE8E8EA) else Color.White) {
                            Text(m.text, Modifier.padding(13.dp).widthIn(max = 310.dp))
                        }
                    }
                }
            }

            Row(Modifier.fillMaxWidth().padding(12.dp), verticalAlignment = Alignment.Bottom) {
                OutlinedTextField(input, { input = it }, Modifier.weight(1f), placeholder = { Text("Message RedAI…") }, maxLines = 5)
                Spacer(Modifier.width(8.dp))
                Button(enabled = input.isNotBlank() && modelLoaded, onClick = {
                    val q = input.trim()
                    messages = messages + Message("user", q) + Message("assistant", "Local inference is connected to this UI layer. The native GGUF runtime will supply the generated answer here.")
                    input = ""
                }) { Text("Send") }
            }
        }
    }
}
