"""
Tests for manuals/services.py: generate_recommended_questions() and its
endpoint (RecommendedQuestionsView). Deterministic and LLM-free - these
never touch Groq, mocked or otherwise.
"""

from django.test import TestCase
from rest_framework.test import APIClient

from manuals.services import generate_recommended_questions

from .helpers import make_chunk, make_machine, make_manual, make_user


class GenerateRecommendedQuestionsTests(TestCase):
    """#10, #11, #12, #13, #14, #15."""

    def test_recommendations_generated_for_the_selected_machine(self):
        """#10, #13: real indexed content produces real, relevant suggestions."""
        machine = make_machine("M-AC90", "AC-90", "Air compressor")
        manual = make_manual(machine, document_id="AC-OM-01")
        make_chunk(manual, "Machine-specific codes. E17 on AC-90 means high discharge temperature.", page_number=2)
        make_chunk(manual, "AF-10 inlet filter restriction indicator must be checked weekly.", page_number=3)

        questions = generate_recommended_questions(machine)
        self.assertTrue(questions)
        self.assertIn("What does E17 indicate?", questions)
        self.assertTrue(any("AF-10 filter" in q for q in questions))

    def test_different_machines_get_different_recommendations(self):
        """#11, #12: machine-specific, and never leaking another machine's content."""
        ac90 = make_machine("M-AC90", "AC-90", "Air compressor")
        hp200 = make_machine("M-HP200", "HP-200", "Hydraulic forming press")
        ac90_manual = make_manual(ac90, document_id="AC-OM-01")
        hp200_manual = make_manual(hp200, document_id="HP-OPS-01")
        make_chunk(ac90_manual, "E17 on AC-90 means high discharge temperature.", page_number=2)
        make_chunk(hp200_manual, "E24 on HP-200 means oil temperature high warning.", page_number=2)

        ac90_questions = generate_recommended_questions(ac90)
        hp200_questions = generate_recommended_questions(hp200)

        self.assertIn("What does E17 indicate?", ac90_questions)
        self.assertNotIn("What does E17 indicate?", hp200_questions)
        self.assertIn("What does E24 indicate?", hp200_questions)
        self.assertNotIn("What does E24 indicate?", ac90_questions)

    def test_duplicate_alarm_code_mentions_produce_one_question_not_many(self):
        """#14: the same code mentioned in several chunks must not duplicate."""
        machine = make_machine()
        manual = make_manual(machine)
        make_chunk(manual, "E17 is a hydraulic pressure fault.", page_number=1)
        make_chunk(manual, "If E17 repeats, escalate to maintenance.", page_number=2)
        make_chunk(manual, "E17 should be logged with the operator.", page_number=3)

        questions = generate_recommended_questions(machine)
        self.assertEqual(questions.count("What does E17 indicate?"), 1)

    def test_no_recommendations_when_machine_has_no_chunks(self):
        """#15: nothing manufactured when there is no suitable source material."""
        machine = make_machine("M-EMPTY", "EMPTY-1", "Unindexed machine")
        self.assertEqual(generate_recommended_questions(machine), [])

    def test_no_recommendations_when_content_matches_no_known_topic(self):
        """#15: generic, pattern-free text yields no suggestions rather than
        inventing a generic/unsupported one."""
        machine = make_machine()
        manual = make_manual(machine)
        make_chunk(manual, "This page intentionally contains only boilerplate legal text and no topics.", page_number=1)
        self.assertEqual(generate_recommended_questions(machine), [])

    def test_result_is_capped_at_the_requested_limit(self):
        machine = make_machine()
        manual = make_manual(machine)
        text = (
            "E17 E24 E31 inspection schedule before each shift record before escalation "
            "READY HOLD troubleshooting observations maintenance schedule"
        )
        make_chunk(manual, text, page_number=1)
        questions = generate_recommended_questions(machine, limit=3)
        self.assertLessEqual(len(questions), 3)


class RecommendedQuestionsViewTests(TestCase):
    """#16: the endpoint requires authentication, matching the app's existing
    read boundary for machine-scoped resources."""

    def setUp(self):
        self.client = APIClient()
        self.machine = make_machine()
        manual = make_manual(self.machine)
        make_chunk(manual, "E17 indicates a hydraulic pressure fault.", page_number=1)

    def test_anonymous_cannot_access_recommendations(self):
        response = self.client.get(f"/api/machines/{self.machine.id}/recommended-questions/")
        self.assertIn(response.status_code, (401, 403))

    def test_authenticated_engineer_gets_machine_specific_questions(self):
        engineer = make_user("eng_reco", "engineer")
        self.client.force_authenticate(engineer)
        response = self.client.get(f"/api/machines/{self.machine.id}/recommended-questions/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("What does E17 indicate?", response.data["questions"])

    def test_unknown_machine_returns_404_not_a_leak(self):
        engineer = make_user("eng_reco_404", "engineer")
        self.client.force_authenticate(engineer)
        response = self.client.get("/api/machines/999999/recommended-questions/")
        self.assertEqual(response.status_code, 404)

    def test_empty_machine_returns_empty_list_not_an_error(self):
        empty_machine = make_machine("M-EMPTY2", "EMPTY-2", "Unindexed machine")
        engineer = make_user("eng_reco_empty", "engineer")
        self.client.force_authenticate(engineer)
        response = self.client.get(f"/api/machines/{empty_machine.id}/recommended-questions/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["questions"], [])
