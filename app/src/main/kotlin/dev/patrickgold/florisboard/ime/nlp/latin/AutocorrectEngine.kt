/*
 * Copyright (C) 2026 The FlorisBoard Contributors
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

import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.read
import kotlin.concurrent.write
import kotlin.math.abs
import kotlin.math.min

/**
 * Offline, fat-finger-aware autocorrect for English (QWERTY).
 *
 * Pure Kotlin, no Android dependencies, no I/O: everything it knows is handed to it in the constructor
 * (the bundled frequency dictionary) and via [learn]/[unlearn] (words the user keeps typing). Nothing here
 * ever touches the network.
 *
 * Scoring = word frequency bonus minus a weighted edit distance where hitting a *neighbouring* key costs much
 * less than hitting a random key. That is what makes `hwllo -> hello` and `yhe -> the` work while leaving
 * unknown-but-intentional words alone.
 *
 * Thread-safe: [process]/[isKnown] run concurrently (suggest + spell), [learn]/[unlearn] take exclusive access.
 */
class AutocorrectEngine(dictionary: Map<String, Int>) {

    data class Scored(val word: String, val score: Double, val cost: Double, val freq: Int)

    data class Result(
        /** What the user typed, untouched. */
        val typed: String,
        /** The auto-applied correction (case-matched), or null if the typed word should be left alone. */
        val correction: String?,
        /** Best alternatives for the suggestion strip, case-matched, never containing [typed] itself. */
        val suggestions: List<String>,
        /** True if [typed] is a known (dictionary or learned) word. */
        val isKnownWord: Boolean,
    )

    private val freq: HashMap<String, Int> = HashMap(dictionary.size * 2)

    /** Dictionary bucketed by stripped length for fast candidate filtering. */
    private val byLength: Array<ArrayList<String>> = Array(MAX_LEN + 1) { ArrayList() }

    /** User-learned words -> how often they were kept. Owned by the caller's storage, mirrored here. */
    private val learned = HashMap<String, Int>()

    /** Guards [byLength] and [learned], which [learn]/[unlearn] mutate while [process] iterates them. */
    private val lock = ReentrantReadWriteLock()

    init {
        for ((word, f) in dictionary) {
            if (word.isEmpty() || !word.all { it.isLetter() || it == '\'' }) continue
            val lw = word.lowercase()
            val stripped = strip(lw)
            if (stripped.length > MAX_LEN) continue
            if (stripped.length == 1 && stripped != "a" && stripped != "i") continue
            val prev = freq[lw]
            if (prev == null || f > prev) freq[lw] = f
            if (prev == null) byLength[stripped.length].add(lw)
        }
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Learning (local only)
    // ---------------------------------------------------------------------------------------------------------------

    fun learn(word: String, weight: Int = 1) {
        val lw = word.lowercase()
        if (lw.length < 2 || lw.length > MAX_LEN || !lw.all { it.isLetter() || it == '\'' }) return
        lock.write {
            val old = learned[lw]
            learned[lw] = (old ?: 0) + weight
            if (old == null && !freq.containsKey(lw)) byLength[strip(lw).length].add(lw)
        }
    }

    fun unlearn(word: String) {
        val lw = word.lowercase()
        lock.write {
            if (learned.remove(lw) != null && !freq.containsKey(lw)) byLength[strip(lw).length].remove(lw)
        }
    }

    fun learnedSnapshot(): Map<String, Int> = lock.read { HashMap(learned) }

    fun loadLearned(data: Map<String, Int>) = lock.write {
        for ((w, c) in data) learn(w, c)
    }

    private fun isTrusted(lw: String): Boolean = (learned[lw] ?: 0) >= TRUST_THRESHOLD

    /** Effective frequency on the 0..255 dictionary scale. Trusted learned words rank like common words. */
    private fun effectiveFreq(lw: String): Int? {
        val d = freq[lw]
        val l = learned[lw]
        if (l == null || l < TRUST_THRESHOLD) return d
        val boosted = min(250, 200 + l * 5)
        return if (d == null) boosted else maxOf(d, boosted)
    }

    fun isKnown(word: String): Boolean {
        val lw = word.lowercase()
        return lock.read { freq.containsKey(lw) || isTrusted(lw) }
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Correction
    // ---------------------------------------------------------------------------------------------------------------

    fun process(typed: String, maxSuggestions: Int = 3): Result = lock.read { processLocked(typed, maxSuggestions) }

    private fun processLocked(typed: String, maxSuggestions: Int): Result {
        if (!isCorrectable(typed)) return Result(typed, null, emptyList(), true)
        val lower = typed.lowercase()
        val stripped = strip(lower)
        val known = isKnown(lower)

        // 1. Hand-curated: common shorthand / transpositions the edit distance can't rank well.
        val direct = when {
            isTrusted(lower) -> null
            stripped == "i" -> "i"
            lower.indexOf('\'') < 0 && CONTRACTIONS.containsKey(stripped) -> CONTRACTIONS[stripped]
            COMMON_TYPOS.containsKey(stripped) -> COMMON_TYPOS[stripped]
            else -> null
        }

        val ranked = rank(stripped, lower, maxSuggestions + 2)
        val suggestions = ranked.asSequence()
            .map { it.word }
            .let { s -> if (direct != null) sequenceOf(direct) + s else s }
            .filter { strip(it) != stripped || it != lower }
            .distinct()
            .take(maxSuggestions)
            .map { matchCase(typed, it) }
            .toList()

        if (direct != null) {
            val corrected = matchCase(typed, direct)
            return Result(typed, corrected.takeIf { it != typed }, suggestions, known)
        }

        val best = ranked.firstOrNull()
        val correction = when {
            best == null -> null
            strip(best.word) == stripped -> null // what they typed is already the best match
            stripped.length < 2 -> null
            !known -> best.word.takeIf { best.cost <= maxCostFor(stripped.length) }
            else -> {
                // Typed a real word: only override when the alternative is vastly more common AND very close.
                val typedFreq = effectiveFreq(lower) ?: 0
                best.word.takeIf { !isTrusted(lower) && best.freq - typedFreq >= 60 + 60 * best.cost && best.cost <= 0.6 }
            }
        }
        return Result(typed, correction?.let { matchCase(typed, it) }, suggestions, known)
    }

    private fun rank(stripped: String, lower: String, k: Int): List<Scored> {
        val n = stripped.length
        if (n == 0 || n > MAX_LEN) return emptyList()
        val maxCost = maxCostFor(n)
        val first = stripped[0]
        val second = if (n > 1) stripped[1] else first
        val out = ArrayList<Scored>()
        val lo = maxOf(1, n - 2)
        val hi = min(MAX_LEN, n + 2)
        for (len in lo..hi) {
            for (cand in byLength[len]) {
                val sc = strip(cand)
                val c0 = sc[0]
                if (c0 != first && c0 != second && !isAdjacent(first, c0)) continue
                val cost = editCost(stripped, sc, maxCost) ?: continue
                val f = effectiveFreq(cand) ?: continue
                out.add(Scored(cand, f / 255.0 * FREQ_WEIGHT - cost, cost, f))
            }
        }
        out.sortByDescending { it.score }
        return if (out.size > k) out.subList(0, k) else out
    }

    /**
     * Weighted Damerau-Levenshtein from [typed] to [word], keyboard-geometry aware. Returns null as soon as the
     * cost is guaranteed to exceed [maxCost] (cheap early exit, keeps a 50k-word scan in the low milliseconds).
     */
    internal fun editCost(typed: String, word: String, maxCost: Double): Double? {
        val n = typed.length
        val m = word.length
        if (abs(n - m) > 2) return null
        var prev2 = DoubleArray(m + 1)
        var prev = DoubleArray(m + 1)
        var cur = DoubleArray(m + 1)
        for (j in 1..m) prev[j] = prev[j - 1] + omitCost(word, j)
        for (i in 1..n) {
            val t = typed[i - 1]
            val extra = if ((i >= 2 && (typed[i - 2] == t || isAdjacent(typed[i - 2], t))) ||
                (i < n && isAdjacent(typed[i], t))) INS_ADJ else INS
            cur[0] = prev[0] + extra
            var rowMin = cur[0]
            for (j in 1..m) {
                val w = word[j - 1]
                var v = if (t == w) prev[j - 1] else prev[j - 1] + (if (isAdjacent(t, w)) SUB_ADJ else SUB)
                v = min(v, prev[j] + extra)
                v = min(v, cur[j - 1] + omitCost(word, j))
                if (i > 1 && j > 1 && t == word[j - 2] && typed[i - 2] == w) v = min(v, prev2[j - 2] + TRANS)
                cur[j] = v
                if (v < rowMin) rowMin = v
            }
            if (rowMin > maxCost) return null
            val tmp = prev2; prev2 = prev; prev = cur; cur = tmp
        }
        return prev[m].takeIf { it <= maxCost }
    }

    private fun omitCost(word: String, j: Int): Double =
        if (j >= 2 && word[j - 1] == word[j - 2]) DEL_DOUBLE else DEL

    companion object {
        const val MAX_LEN = 24
        const val TRUST_THRESHOLD = 2

        // Cost weights (tuned against the bundled dictionary, see AutocorrectEngineTest)
        const val SUB_ADJ = 0.5   // hit a neighbouring key
        const val SUB = 1.0       // hit an unrelated key
        const val TRANS = 0.6     // swapped two letters
        const val INS_ADJ = 0.55  // extra letter from a neighbour / double tap
        const val INS = 1.0
        const val DEL = 0.85      // missed a letter
        const val DEL_DOUBLE = 0.4 // missed one half of a double letter
        const val FREQ_WEIGHT = 1.5

        fun maxCostFor(len: Int): Double = when {
            len <= 3 -> 0.5
            len <= 5 -> 1.0
            len <= 8 -> 1.5
            else -> 2.0
        }

        private val ROWS = arrayOf("qwertyuiop", "asdfghjkl", "zxcvbnm")
        private val ROW = IntArray(26) { -1 }
        private val COL = IntArray(26) { -1 }

        init {
            for ((r, row) in ROWS.withIndex()) for ((c, ch) in row.withIndex()) {
                ROW[ch - 'a'] = r; COL[ch - 'a'] = c
            }
        }

        /** True if [a] and [b] touch each other on a standard staggered QWERTY layout. */
        fun isAdjacent(a: Char, b: Char): Boolean {
            if (a == b || a !in 'a'..'z' || b !in 'a'..'z') return false
            val r1 = ROW[a - 'a']; val c1 = COL[a - 'a']
            val r2 = ROW[b - 'a']; val c2 = COL[b - 'a']
            return when (r2 - r1) {
                0 -> abs(c1 - c2) == 1
                -1 -> c2 == c1 || c2 == c1 + 1 // b is in the row above a
                1 -> c2 == c1 || c2 == c1 - 1  // b is in the row below a
                else -> false
            }
        }

        fun strip(w: String): String = if (w.indexOf('\'') < 0) w else w.replace("'", "")

        /** Only plain alphabetic words are touched. Anything with digits, @, /, etc. is left alone. */
        fun isCorrectable(word: String): Boolean {
            if (word.isEmpty() || word.length > MAX_LEN) return false
            if (!word.all { it in 'a'..'z' || it in 'A'..'Z' || it == '\'' }) return false
            if (!word[0].isLetter()) return false
            // Intentional mixed case like "iPhone" or "McDonald" -> leave it.
            val upperAfterFirst = word.drop(1).count { it.isUpperCase() }
            if (upperAfterFirst in 1 until word.length - 1) return false
            return true
        }

        fun matchCase(typed: String, word: String): String {
            if (word == "i") return "I"
            if (word.startsWith("i'")) return "I" + word.substring(1)
            val letters = typed.filter { it.isLetter() }
            return when {
                letters.length > 1 && letters.all { it.isUpperCase() } -> word.uppercase()
                typed.firstOrNull()?.isUpperCase() == true -> word.replaceFirstChar { it.uppercaseChar() }
                else -> word
            }
        }

        val CONTRACTIONS: Map<String, String> = mapOf(
            "im" to "i'm", "ive" to "i've", "dont" to "don't", "cant" to "can't", "wont" to "won't",
            "didnt" to "didn't", "doesnt" to "doesn't", "isnt" to "isn't", "arent" to "aren't", "wasnt" to "wasn't",
            "werent" to "weren't", "hasnt" to "hasn't", "havent" to "haven't", "hadnt" to "hadn't",
            "wouldnt" to "wouldn't", "couldnt" to "couldn't", "shouldnt" to "shouldn't", "mustnt" to "mustn't",
            "thats" to "that's", "whats" to "what's", "youre" to "you're", "theyre" to "they're",
            "youve" to "you've", "theyve" to "they've", "weve" to "we've", "youll" to "you'll",
            "theyll" to "they'll", "itll" to "it'll", "thatll" to "that'll", "aint" to "ain't",
            "heres" to "here's", "theres" to "there's", "wheres" to "where's", "whos" to "who's",
            "hows" to "how's", "shes" to "she's", "youd" to "you'd", "theyd" to "they'd",
            "wouldve" to "would've", "couldve" to "could've", "shouldve" to "should've", "yall" to "y'all",
            "idk" to "idk", "itd" to "it'd",
        ).filter { it.key != it.value }

        val COMMON_TYPOS: Map<String, String> = mapOf(
            "teh" to "the", "hte" to "the", "yhe" to "the", "tge" to "the", "rhe" to "the", "thw" to "the",
            "adn" to "and", "nad" to "and", "abd" to "and", "anf" to "and", "amd" to "and",
            "taht" to "that", "thta" to "that", "thier" to "their", "recieve" to "receive",
            "recieved" to "received", "definately" to "definitely", "definatly" to "definitely",
            "alot" to "a lot", "becuase" to "because", "beacuse" to "because", "jsut" to "just",
            "waht" to "what", "wiht" to "with", "whith" to "with", "freind" to "friend", "freinds" to "friends",
            "beleive" to "believe", "acheive" to "achieve", "wierd" to "weird", "untill" to "until",
            "tommorow" to "tomorrow", "tomorow" to "tomorrow", "seperate" to "separate", "occured" to "occurred",
            "goverment" to "government", "enviroment" to "environment", "begining" to "beginning",
            "truely" to "truly", "finaly" to "finally", "realy" to "really", "basicly" to "basically",
            "wich" to "which", "knwo" to "know", "konw" to "know", "nkow" to "know", "thnk" to "think",
            "ahve" to "have", "hvae" to "have", "cna" to "can", "nto" to "not", "fomr" to "from",
            "ger" to "get", "yuo" to "you", "yoi" to "you", "ot" to "to",
            "fo" to "of", "si" to "is", "ti" to "it", "fro" to "for", "agian" to "again", "aswell" to "as well",
            "wnat" to "want", "watn" to "want", "dosent" to "doesn't", "doesnt" to "doesn't",
            "youre" to "you're", "probaly" to "probably", "proably" to "probably",
        )
    }
}
