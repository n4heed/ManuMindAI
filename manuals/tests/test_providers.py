from django.test import TestCase

from manuals.rag.providers import LocalEmbeddingProvider, LocalLLMProvider, cosine_similarity


class LocalEmbeddingProviderTests(TestCase):
    def setUp(self):
        self.embedder = LocalEmbeddingProvider()

    def test_embedding_is_deterministic(self):
        [v1] = self.embedder.embed(["normal operating pressure range"])
        [v2] = self.embedder.embed(["normal operating pressure range"])
        self.assertEqual(v1, v2)

    def test_similar_text_scores_higher_than_unrelated_text(self):
        [question] = self.embedder.embed(["What is the normal operating pressure range?"])
        [related] = self.embedder.embed(["The normal operating pressure range is 80 to 120 PSI."])
        [unrelated] = self.embedder.embed(["Replace the gasket every twelve months during inspection."])

        related_score = cosine_similarity(question, related)
        unrelated_score = cosine_similarity(question, unrelated)
        self.assertGreater(related_score, unrelated_score)

    def test_empty_text_gives_empty_vector(self):
        [v] = self.embedder.embed([""])
        self.assertEqual(v, {})

    def test_light_stemming_matches_different_word_forms(self):
        # "recorded" (question) vs "record" (manual text) - different exact strings,
        # same stem. Without stemming this pair scores 0.0 (see README section 3).
        [q] = self.embedder.embed(["What should be recorded before escalation?"])
        [c] = self.embedder.embed(["Record the alarm code before escalating the fault."])
        self.assertGreater(cosine_similarity(q, c), 0.0)

    def test_stemming_does_not_merge_unrelated_short_words(self):
        # Guard against over-aggressive stripping, e.g. "process"/"bus" should not
        # become "proce"/"bu" and accidentally collide with something else.
        [v] = self.embedder.embed(["process"])
        self.assertIn("process", v)


class LocalLLMProviderTests(TestCase):
    def test_extractive_answer_uses_relevant_sentence(self):
        llm = LocalLLMProvider()
        answer = llm.generate(
            "How often is the filter checked?",
            [{"text": "The AF-10 filter indicator is checked weekly. The housing is inspected monthly.",
              "document_id": "AC-OM-01", "page_number": 2}],
        )
        self.assertIn("weekly", answer.lower())

    def test_is_marked_as_deterministic_fallback(self):
        self.assertTrue(LocalLLMProvider().is_deterministic_fallback)
