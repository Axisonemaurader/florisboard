package dev.patrickgold.florisboard.ime.nlp.latin

import kotlinx.serialization.builtins.MapSerializer
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.json.Json
import java.io.File
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class AutocorrectEngineTest {
    private val engine: AutocorrectEngine by lazy {
        val f = listOf("src/main/assets/ime/dict/data.json", "app/src/main/assets/ime/dict/data.json")
            .map(::File).first { it.exists() }
        AutocorrectEngine(Json.decodeFromString(MapSerializer(String.serializer(), Int.serializer()), f.readText()))
    }

    private fun corr(w: String) = engine.process(w).correction

    @Test
    fun adjacency() {
        assertTrue(AutocorrectEngine.isAdjacent('g', 'h'))
        assertTrue(AutocorrectEngine.isAdjacent('a', 'q'))
        assertTrue(AutocorrectEngine.isAdjacent('z', 's'))
        assertTrue(AutocorrectEngine.isAdjacent('h', 'y'))
        assertFalse(AutocorrectEngine.isAdjacent('a', 'e'))
        assertFalse(AutocorrectEngine.isAdjacent('q', 'p'))
    }

    @Test
    fun fatFingerTypos() {
        val cases = mapOf(
            "hwllo" to "hello", "teh" to "the", "yhe" to "the", "tge" to "the", "becuase" to "because",
            "thsi" to "this", "wirk" to "work", "ypu" to "you", "whst" to "what", "goung" to "going",
            "tomorow" to "tomorrow", "recieve" to "receive", "keybaord" to "keyboard", "sonething" to "something",
            "probaly" to "probably", "realy" to "really", "wjat" to "what", "tjis" to "this", "hpw" to "how",
            "abd" to "and", "ger" to "get", "nkt" to "not", "liek" to "like", "thikn" to "think",
            "pepole" to "people", "bevause" to "because", "woek" to "work", "yiu" to "you", "frist" to "first",
        )
        val misses = cases.filter { (typed, want) -> corr(typed) != want }
            .map { (typed, want) -> "$typed -> ${corr(typed)} (want $want)" }
        assertTrue(misses.isEmpty(), "Missed: $misses")
    }

    @Test
    fun contractionsAndI() {
        assertEquals("don't", corr("dont"))
        assertEquals("I'm", corr("im"))
        assertEquals("I", corr("i"))
        assertEquals("Don't", corr("Dont"))
    }

    @Test
    fun leavesRealWordsAlone() {
        for (w in listOf("hello", "the", "keyboard", "work", "were", "well", "hell", "its", "form", "lol", "ok")) {
            assertNull(corr(w), "should not correct '$w'")
        }
    }

    @Test
    fun leavesNonWordsAlone() {
        for (w in listOf("abc123", "user@mail", "iPhone", "http", "")) {
            assertNull(corr(w), "should not correct '$w'")
        }
    }

    @Test
    fun casePreserved() {
        assertEquals("Hello", corr("Hwllo"))
        assertEquals("HELLO", corr("HWLLO"))
    }

    @Test
    fun learningStopsCorrection() {
        val e = engine
        e.learn("zakk", AutocorrectEngine.TRUST_THRESHOLD)
        assertNull(e.process("zakk").correction)
        e.unlearn("zakk")
    }

    /**
     * Regression: suggest() runs on Dispatchers.Default while notifySuggestionAccepted/Reverted call learn() from
     * another thread. Without locking, rank() threw ConcurrentModificationException iterating byLength[len].
     */
    @Test
    fun learnWhileProcessingIsThreadSafe() {
        val e = engine
        val failures = ConcurrentLinkedQueue<Throwable>()
        val stop = AtomicBoolean(false)
        // Same stripped lengths as "sonething" so every learn() lands in a bucket rank() is scanning.
        fun word(i: Int) = "sonet" + (0 until 4).map { 'a' + (i shr (it * 4)) % 16 }.joinToString("")

        val readers = List(4) {
            thread {
                try {
                    while (!stop.get()) {
                        e.process("sonething")
                        e.isKnown("sonethinx")
                    }
                } catch (t: Throwable) {
                    failures.add(t)
                }
            }
        }
        val writer = thread {
            try {
                for (i in 0 until 20_000) {
                    e.learn(word(i))
                    if (i % 3 == 0) e.unlearn(word(i / 2))
                    if (i % 500 == 0) e.learnedSnapshot()
                }
            } catch (t: Throwable) {
                failures.add(t)
            } finally {
                stop.set(true)
            }
        }
        writer.join(60_000)
        stop.set(true)
        readers.forEach { it.join(10_000) }

        assertTrue(failures.isEmpty(), "Concurrent access failed: ${failures.firstOrNull()}")
        assertEquals("something", e.process("sonething").correction)
    }

    @Test
    fun fastEnough() {
        engine.process("warmup")
        val start = System.nanoTime()
        repeat(50) { engine.process("sonething") }
        val perCallMs = (System.nanoTime() - start) / 50 / 1_000_000.0
        assertTrue(perCallMs < 50.0, "too slow: ${perCallMs}ms per word")
    }
}
