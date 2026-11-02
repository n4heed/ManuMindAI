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
