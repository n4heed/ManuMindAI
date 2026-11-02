from django.test import TestCase

from manuals.services import CHUNK_TARGET_CHARS, chunk_page_text


class ChunkPageTextTests(TestCase):
    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(chunk_page_text(""), [])
        self.assertEqual(chunk_page_text("   \n  "), [])

    def test_short_text_is_a_single_chunk(self):
        text = "The AF-10 filter should be checked weekly."
        chunks = chunk_page_text(text)
        self.assertEqual(len(chunks), 1)
        self.assertIn("AF-10", chunks[0])

    def test_long_text_is_split_into_multiple_chunks(self):
        paragraph = "This sentence describes a maintenance step in detail. " * 40
        text = "\n\n".join([paragraph, paragraph, paragraph])
        chunks = chunk_page_text(text)
        self.assertGreater(len(chunks), 1)
        # Overlap is intentional, but chunks shouldn't balloon far past the target size.
        for chunk in chunks:
            self.assertLess(len(chunk), CHUNK_TARGET_CHARS * 2)

    def test_consecutive_chunks_overlap(self):
        paragraph_a = "Alpha step. " * 60
        paragraph_b = "Bravo step. " * 60
        text = f"{paragraph_a}\n\n{paragraph_b}"
        chunks = chunk_page_text(text)
        self.assertGreaterEqual(len(chunks), 2)
        # The tail of chunk N should reappear at the start of chunk N+1.
        tail_fragment = chunks[0][-20:]
        self.assertIn(tail_fragment, chunks[1])
