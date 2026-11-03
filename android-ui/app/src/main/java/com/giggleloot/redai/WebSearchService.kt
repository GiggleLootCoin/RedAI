package com.giggleloot.redai

import android.content.Context
import android.net.Uri
import android.util.Base64
import android.util.Log
import android.text.Html
import org.json.JSONObject
import org.jsoup.Jsoup
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

data class WebSource(val title: String, val url: String, val excerpt: String, val date: String = "")

data class WebSearchResult(
    val sources: List<WebSource>,
    val context: String,
    val error: String? = null
) {
    val succeeded: Boolean get() = sources.isNotEmpty()
    fun sourceList(): String = sources.mapIndexed { i, s ->
        "${i + 1}. ${s.title}\n${s.url}" + if (s.date.isNotBlank()) "\nDate: ${s.date}" else ""
    }.joinToString("\n\n")
}

class WebSearchService(context: Context) {
    private val prefs = context.getSharedPreferences("redai_web_secure", Context.MODE_PRIVATE)
    private val keyAlias = "RedAIWebSearchKey"

    fun saveTavilyKey(value: String) {
        if (value.isBlank()) {
            prefs.edit().remove("tavily_key").remove("tavily_iv").apply()
            return
        }
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, getOrCreateKey())
        val encrypted = cipher.doFinal(value.trim().toByteArray(Charsets.UTF_8))
        prefs.edit()
            .putString("tavily_iv", Base64.encodeToString(cipher.iv, Base64.NO_WRAP))
            .putString("tavily_key", Base64.encodeToString(encrypted, Base64.NO_WRAP))
            .apply()
    }

    fun hasTavilyKey(): Boolean =
        !prefs.getString("tavily_key", null).isNullOrBlank() &&
            !prefs.getString("tavily_iv", null).isNullOrBlank()

    private fun readTavilyKey(): String? = runCatching {
        val encrypted = Base64.decode(prefs.getString("tavily_key", null), Base64.NO_WRAP)
        val iv = Base64.decode(prefs.getString("tavily_iv", null), Base64.NO_WRAP)
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.DECRYPT_MODE, getOrCreateKey(), GCMParameterSpec(128, iv))
        String(cipher.doFinal(encrypted), Charsets.UTF_8)
    }.getOrNull()

    private fun getOrCreateKey(): SecretKey {
        val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        val existing = store.getKey(keyAlias, null)
        if (existing is SecretKey) return existing
        val generator = KeyGenerator.getInstance("AES", "AndroidKeyStore")
        generator.init(
            android.security.keystore.KeyGenParameterSpec.Builder(
                keyAlias,
                android.security.keystore.KeyProperties.PURPOSE_ENCRYPT or
                    android.security.keystore.KeyProperties.PURPOSE_DECRYPT
            ).setBlockModes(android.security.keystore.KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(android.security.keystore.KeyProperties.ENCRYPTION_PADDING_NONE)
                .setRandomizedEncryptionRequired(true)
                .build()
        )
        return generator.generateKey()
    }

    fun search(query: String, provider: String): WebSearchResult {
        if (query.isBlank()) return WebSearchResult(emptyList(), "", "The search query was empty.")
        return try {
            when (provider) {
                "tavily" -> {
                    val key = readTavilyKey()
                        ?: return WebSearchResult(emptyList(), "", "Tavily is selected but no API key is saved. Choose DuckDuckGo or Wikipedia, or add your own free Tavily key in RedAI settings.")
                    searchTavily(query, key)
                }
                "wikipedia" -> searchWikipedia(query)
                else -> {
                    val web = searchDuckDuckGo(query)
                    if (web.succeeded) web else {
                        val wiki = searchWikipedia(query)
                        if (wiki.succeeded) wiki else WebSearchResult(emptyList(), "", web.error ?: wiki.error)
                    }
                }
            }
        } catch (e: Exception) {
            Log.w("RedAIWebSearch", "Search failed: ${e.javaClass.simpleName}")
            WebSearchResult(emptyList(), "", "Live search failed (${e.javaClass.simpleName}). RedAI did not receive search results.")
        }
    }

    private fun searchWikipedia(query: String): WebSearchResult {
        val encoded = URLEncoder.encode(query.take(400), "UTF-8")
        val connection = (URL("https://en.wikipedia.org/w/rest.php/v1/search/page?q=$encoded&limit=5").openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 6000
            readTimeout = 8000
            setRequestProperty("User-Agent", "RedAI-Android/3.2 (offline-first assistant; search integration)")
            setRequestProperty("Accept", "application/json")
        }
        try {
            val code = connection.responseCode
            if (code !in 200..299) return WebSearchResult(emptyList(), "", "Wikipedia search returned HTTP $code.")
            val body = connection.inputStream.bufferedReader().use { it.readText().take(500_000) }
            val pages = JSONObject(body).optJSONArray("pages") ?: return WebSearchResult(emptyList(), "", "Wikipedia returned no matching pages.")
            val sources = (0 until minOf(pages.length(), 5)).mapNotNull { i ->
                val page = pages.optJSONObject(i) ?: return@mapNotNull null
                val key = page.optString("key")
                if (key.isBlank()) return@mapNotNull null
                val link = "https://en.wikipedia.org/wiki/" + Uri.encode(key.replace(" ", "_"))
                WebSource(
                    clean(page.optString("title").ifBlank { key }),
                    link,
                    clean(page.optString("excerpt").ifBlank { page.optString("description") }).take(1600)
                )
            }
            val enriched = enrichSources(sources)
            return WebSearchResult(enriched, formatContext(enriched))
        } finally {
            connection.disconnect()
        }
    }

    private fun searchTavily(query: String, key: String): WebSearchResult {
        val connection = (URL("https://api.tavily.com/search").openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = 7000
            readTimeout = 9000
            doOutput = true
            setRequestProperty("Content-Type", "application/json")
            setRequestProperty("Accept", "application/json")
            setRequestProperty("Authorization", "Bearer $key")
            setRequestProperty("User-Agent", "RedAI-Android/3.2")
        }
        try {
            val payload = JSONObject().put("query", query.take(500))
                .put("search_depth", "basic")
                .put("max_results", 5)
                .put("include_answer", false)
                .put("include_raw_content", false)
            connection.outputStream.use { it.write(payload.toString().toByteArray(Charsets.UTF_8)) }
            val code = connection.responseCode
            val stream = if (code in 200..299) connection.inputStream else connection.errorStream
            val body = stream?.bufferedReader()?.use { it.readText().take(500_000) }.orEmpty()
            if (code !in 200..299) return WebSearchResult(emptyList(), "", "Tavily returned HTTP $code. Check the key or monthly free allowance.")
            val array = JSONObject(body).optJSONArray("results") ?: return WebSearchResult(emptyList(), "", "Tavily returned no readable results.")
            val sources = (0 until minOf(array.length(), 5)).mapNotNull { i ->
                val item = array.optJSONObject(i) ?: return@mapNotNull null
                val link = item.optString("url")
                if (!isSafePublicUrl(link)) return@mapNotNull null
                WebSource(
                    clean(item.optString("title").ifBlank { link }),
                    link,
                    clean(item.optString("content")).take(1800),
                    item.optString("published_date")
                )
            }
            val enriched = enrichSources(sources)
            return WebSearchResult(enriched, formatContext(enriched))
        } finally {
            connection.disconnect()
        }
    }

    private fun searchDuckDuckGo(query: String): WebSearchResult {
        val encoded = URLEncoder.encode(query.take(500), "UTF-8")
        val connection = (URL("https://html.duckduckgo.com/html/?q=$encoded").openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 6500
            readTimeout = 8500
            setRequestProperty("User-Agent", "Mozilla/5.0 (Linux; Android 12) AppleWebKit/537.36 Chrome/120 Mobile Safari/537.36")
            setRequestProperty("Accept", "text/html")
        }
        try {
            val code = connection.responseCode
            if (code !in 200..299) return WebSearchResult(emptyList(), "", "DuckDuckGo search returned HTTP $code. Try again later or select Tavily.")
            val html = connection.inputStream.bufferedReader().use { it.readText().take(1_500_000) }
            val anchorRegex = Regex("(?is)<a\\b(?=[^>]*class=['\"][^'\"]*result__a)[^>]*>(.*?)</a>")
            val sources = anchorRegex.findAll(html).mapNotNull { match ->
                val tagStart = html.lastIndexOf("<a", match.range.first, ignoreCase = true)
                val tagEnd = html.indexOf(">", tagStart).takeIf { it >= 0 } ?: return@mapNotNull null
                val tag = html.substring(tagStart, tagEnd + 1)
                val href = Regex("(?is)href=['\"]([^'\"]+)['\"]").find(tag)?.groupValues?.get(1) ?: return@mapNotNull null
                val decoded = runCatching {
                    val parsed = Uri.parse(href)
                    parsed.getQueryParameter("uddg") ?: href
                }.getOrDefault(href)
                val link = Uri.decode(decoded)
                if (!isSafePublicUrl(link)) return@mapNotNull null
                val title = clean(match.groupValues[1])
                val following = html.substring(match.range.last + 1, minOf(html.length, match.range.last + 900))
                val snippetHtml = Regex("(?is)<(?:a|div)[^>]*class=['\"][^'\"]*result__snippet[^'\"]*['\"][^>]*>(.*?)</(?:a|div)>")
                    .find(following)?.groupValues?.get(1).orEmpty()
                WebSource(title.ifBlank { link }, link, clean(snippetHtml).take(1600))
            }.distinctBy { it.url }.take(5).toList()
            val enriched = enrichSources(sources)
            if (enriched.isEmpty()) return WebSearchResult(emptyList(), "", "DuckDuckGo returned no extractable results. The search page may have changed or blocked automated access.")
            return WebSearchResult(enriched, formatContext(enriched))
        } finally {
            connection.disconnect()
        }
    }

    private fun enrichSources(sources: List<WebSource>): List<WebSource> =
        sources.mapIndexed { index, source ->
            if (index >= 2 || source.excerpt.length >= 900) source
            else source.copy(excerpt = fetchPageExcerpt(source.url).ifBlank { source.excerpt })
        }

    private fun fetchPageExcerpt(link: String): String {
        if (!isSafePublicUrl(link)) return ""
        var current = link
        repeat(4) {
            val url = URL(current)
            if (!isSafePublicUrl(current)) return ""
            val c = (url.openConnection() as HttpURLConnection).apply {
                instanceFollowRedirects = false
                connectTimeout = 3500
                readTimeout = 4500
                setRequestProperty("User-Agent", "RedAI-Android/3.2 (mobile web research)")
                setRequestProperty("Accept", "text/html")
            }
            try {
                val code = c.responseCode
                if (code in 300..399) {
                    val location = c.getHeaderField("Location") ?: return ""
                    current = URL(url, location).toString()
                    if (!isSafePublicUrl(current)) return ""
                    return@repeat
                }
                if (code !in 200..299 || !c.contentType.orEmpty().contains("text/html", true)) return ""
                val bytes = c.inputStream.use { input ->
                    val output = java.io.ByteArrayOutputStream()
                    val buffer = ByteArray(8192)
                    var total = 0
                    while (total < 350_000) {
                        val n = input.read(buffer, 0, minOf(buffer.size, 350_000 - total))
                        if (n <= 0) break
                        output.write(buffer, 0, n)
                        total += n
                    }
                    output.toByteArray()
                }
                val doc = Jsoup.parse(String(bytes, Charsets.UTF_8), current)
                doc.select("script,style,nav,footer,header,form,aside,noscript,svg").remove()
                return clean(doc.body()?.text().orEmpty()).take(2200)
            } finally {
                c.disconnect()
            }
        }
        return ""
    }

    private fun isSafePublicUrl(value: String): Boolean = runCatching {
        val uri = Uri.parse(value)
        val host = uri.host?.lowercase() ?: return false
        if (uri.scheme != "https" || uri.userInfo != null) return false
        if (host == "localhost" || host.endsWith(".local") || host.endsWith(".internal")) return false
        if (host == "127.0.0.1" || host == "::1" || host.startsWith("10.") || host.startsWith("192.168.") || host.startsWith("169.254.")) return false
        val parts = host.split(".")
        if (parts.size == 4 && parts.all { part -> (part.toIntOrNull() ?: -1) in 0..255 }) {
            val a = parts[0].toInt()
            val b = parts[1].toInt()
            if (a == 0 || a == 10 || a == 127 || (a == 172 && b in 16..31) || (a == 192 && b == 168) || (a == 169 && b == 254)) return false
        }
        true
    }.getOrDefault(false)

    private fun clean(value: String): String =
        Html.fromHtml(value, Html.FROM_HTML_MODE_LEGACY).toString()
            .replace(Regex("\\s+"), " ").trim()

    private fun formatContext(sources: List<WebSource>): String = sources.mapIndexed { i, s ->
        "[SOURCE ${i + 1}] ${s.title}\nURL: ${s.url}" +
            (if (s.date.isNotBlank()) "\nPublished: ${s.date}" else "") +
            "\nExcerpt: ${s.excerpt}"
    }.joinToString("\n\n").take(8500)
}
