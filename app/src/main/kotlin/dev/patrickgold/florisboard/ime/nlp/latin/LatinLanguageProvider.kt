/*
 * Copyright (C) 2022-2025 The FlorisBoard Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package dev.patrickgold.florisboard.ime.nlp.latin

import android.content.Context
import androidx.compose.ui.graphics.vector.ImageVector
import dev.patrickgold.florisboard.appContext
import dev.patrickgold.florisboard.ime.core.Subtype
import dev.patrickgold.florisboard.ime.editor.EditorContent
import dev.patrickgold.florisboard.ime.nlp.SpellingProvider
import dev.patrickgold.florisboard.ime.nlp.SpellingResult
import dev.patrickgold.florisboard.ime.nlp.SuggestionCandidate
import dev.patrickgold.florisboard.ime.nlp.SuggestionProvider
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.builtins.MapSerializer
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.json.Json
import org.florisboard.lib.android.readText
import org.florisboard.lib.kotlin.guardedByLock
import java.io.File

/**
 * Candidate produced by the autocorrect engine. [typed] is what the user actually typed, so a revert (backspace
 * right after an auto-correction) can restore it and teach the engine to leave that word alone.
 */
data class AutocorrectCandidate(
    override val text: CharSequence,
    val typed: String,
    val isKeepTyped: Boolean,
    override val confidence: Double,
    override val isEligibleForAutoCommit: Boolean,
    override val sourceProvider: SuggestionProvider?,
) : SuggestionCandidate {
    override val secondaryText: CharSequence? = null
    override val isEligibleForUserRemoval: Boolean = false
    override val icon: ImageVector? = null
}

/**
 * Offline English autocorrect + suggestions.
 *
 * PRIVACY: everything is local. The only file written is the learned-word list in `noBackupFilesDir`, which is
 * excluded from Android cloud backup and device transfer by definition. Nothing is ever sent anywhere (the app
 * does not hold the INTERNET permission).
 */
class LatinLanguageProvider(context: Context) : SpellingProvider, SuggestionProvider {
    companion object {
        // Default user ID used for all subtypes, unless otherwise specified.
        // See `ime/core/Subtype.kt` Line 210 and 211 for the default usage
        const val ProviderId = "org.florisboard.nlp.providers.latin"

        private const val LEARNED_FILE = "autocorrect_learned.json"
        private const val MAX_LEARNED = 5000
    }

    private val appContext by context.appContext()
    private val ioScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    private val wordData = guardedByLock { mutableMapOf<String, Int>() }
    private val wordDataSerializer = MapSerializer(String.serializer(), Int.serializer())

    @Volatile
    private var engine: AutocorrectEngine? = null

    override val providerId = ProviderId

    override suspend fun create() {
        // Nothing language-independent to set up.
    }

    private fun learnedFile(): File = File(appContext.noBackupFilesDir, LEARNED_FILE)

    override suspend fun preload(subtype: Subtype) = withContext(Dispatchers.IO) {
        wordData.withLock { wordData ->
            if (wordData.isEmpty()) {
                val rawData = appContext.assets.readText("ime/dict/data.json")
                val jsonData = Json.decodeFromString(wordDataSerializer, rawData)
                wordData.putAll(jsonData)
            }
            if (engine == null) {
                val e = AutocorrectEngine(wordData)
                runCatching {
                    val f = learnedFile()
                    if (f.exists()) e.loadLearned(Json.decodeFromString(wordDataSerializer, f.readText()))
                }
                engine = e
            }
        }
    }

    private fun persistLearned() {
        val e = engine ?: return
        ioScope.launch {
            runCatching {
                val snapshot = e.learnedSnapshot().entries
                    .sortedByDescending { it.value }
                    .take(MAX_LEARNED)
                    .associate { it.key to it.value }
                val f = learnedFile()
                val tmp = File(f.parentFile, "$LEARNED_FILE.tmp")
                tmp.writeText(Json.encodeToString(wordDataSerializer, snapshot))
                tmp.renameTo(f)
            }
        }
    }

    override suspend fun spell(
        subtype: Subtype,
        word: String,
        precedingWords: List<String>,
        followingWords: List<String>,
        maxSuggestionCount: Int,
        allowPossiblyOffensive: Boolean,
        isPrivateSession: Boolean,
    ): SpellingResult {
        val e = engine ?: return SpellingResult.unspecified()
        if (!AutocorrectEngine.isCorrectable(word) || e.isKnown(word)) return SpellingResult.validWord()
        val result = e.process(word, maxSuggestionCount.coerceAtLeast(1))
        return if (result.suggestions.isEmpty()) SpellingResult.validWord()
        else SpellingResult.typo(result.suggestions.toTypedArray())
    }

    override suspend fun suggest(
        subtype: Subtype,
        content: EditorContent,
        maxCandidateCount: Int,
        allowPossiblyOffensive: Boolean,
        isPrivateSession: Boolean,
    ): List<SuggestionCandidate> {
        val e = engine ?: return emptyList()
        val typed = content.composingText.ifBlank { content.currentWordText }
        if (typed.isBlank() || !AutocorrectEngine.isCorrectable(typed)) return emptyList()
        val result = e.process(typed, maxSuggestions = (maxCandidateCount - 1).coerceIn(1, 4))
        return buildList {
            val correction = result.correction
            if (correction != null) {
                add(AutocorrectCandidate(correction, typed, false, 0.9, true, this@LatinLanguageProvider))
                // Let the user tap their original word to keep it (and teach the engine).
                add(AutocorrectCandidate(typed, typed, true, 0.5, false, this@LatinLanguageProvider))
            }
            for (s in result.suggestions) {
                if (s == correction || s == typed) continue
                add(AutocorrectCandidate(s, typed, false, 0.4, false, this@LatinLanguageProvider))
            }
        }.take(maxCandidateCount)
    }

    override suspend fun notifySuggestionAccepted(subtype: Subtype, candidate: SuggestionCandidate) {
        val c = candidate as? AutocorrectCandidate ?: return
        val e = engine ?: return
        if (c.isKeepTyped) {
            // User explicitly chose their own spelling: trust it immediately.
            e.learn(c.typed, AutocorrectEngine.TRUST_THRESHOLD)
            persistLearned()
        }
    }

    override suspend fun notifySuggestionReverted(subtype: Subtype, candidate: SuggestionCandidate) {
        val c = candidate as? AutocorrectCandidate ?: return
        val e = engine ?: return
        // Backspace right after an auto-correction = "I meant what I typed". Never correct it again.
        e.learn(c.typed, AutocorrectEngine.TRUST_THRESHOLD)
        persistLearned()
    }

    override suspend fun removeSuggestion(subtype: Subtype, candidate: SuggestionCandidate): Boolean {
        return false
    }

    override suspend fun getListOfWords(subtype: Subtype): List<String> {
        return wordData.withLock { it.keys.toList() }
    }

    override suspend fun getFrequencyForWord(subtype: Subtype, word: String): Double {
        return wordData.withLock { it.getOrDefault(word, 0) / 255.0 }
    }

    override suspend fun destroy() {
        // Process is usually killed before this is called.
    }
}
