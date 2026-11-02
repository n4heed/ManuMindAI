"""
Optional, manual-only live smoke test for the Groq integration.

This makes REAL network calls to the configured Groq API and is never run
automatically by this project's test suite, CI, or any other command. Run it
yourself, after setting LLM_ENABLED=True and a real GROQ_API_KEY in your .env:

    python manage.py groq_smoke_test

Exercises four scenarios against the real structured-output API: a simple
factual question (one claim, one source), a multi-part question (expects
more than one claim), a question needing multiple passages for one claim,
and an insufficient-evidence question (expects status=insufficient_evidence,
no claims). See README "Dual-Answer RAG" for the full explanation.
"""

import sys

from django.core.management.base import BaseCommand, CommandError

from manuals.rag.groq_provider import GroqConfigurationError, GroqProviderError, get_groq_provider


def _console_safe(text: str) -> str:
    """
    Groq's real responses can contain Unicode punctuation (smart quotes, en/em
    dashes, non-breaking hyphens) that a Windows console's legacy codepage
    (e.g. cp1252) cannot encode, which would otherwise crash this command
    with a UnicodeEncodeError after a genuinely successful API call. Falls
    back to '?' for anything the current stdout encoding can't display;
    never affects what was actually returned by Groq, only this printout.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


FILTER_SOURCE = {
    "id": "S1",
    "document_id": "SMOKE-TEST-MANUAL",
    "page_number": 1,
    "text": "The AF-10 filter indicator must be checked every 100 operating hours. "
    "Planned replacement of the AF-10 filter is required every 600 operating hours.",
}
ESCALATION_SOURCE = {
    "id": "S2",
    "document_id": "SMOKE-TEST-MANUAL",
    "page_number": 2,
    "text": "Before escalating a fault, record the machine ID, timestamp, active alarm code, and "
    "displayed pressure.",
}

SCENARIOS = [
    (
        "Simple factual question (expect: answered, 1 claim, source S1)",
        "How often should the AF-10 filter be checked?",
        [FILTER_SOURCE],
    ),
    (
        "Multi-part question (expect: answered, 2 claims, one per source)",
        "How often should the AF-10 filter be checked, and what should be recorded before escalating a fault?",
        [FILTER_SOURCE, ESCALATION_SOURCE],
    ),
    (
        "Question needing multiple passages for one claim (expect: answered, "
        "a claim citing both S1 and S2)",
        "Summarize the filter check interval and the escalation recording requirement together in one statement.",
        [FILTER_SOURCE, ESCALATION_SOURCE],
    ),
    (
        "Insufficient-evidence question (expect: status=insufficient_evidence, 0 claims)",
        "What is the warranty claim procedure for third-party engines?",
        [FILTER_SOURCE],
    ),
]


class Command(BaseCommand):
    help = (
        "Makes REAL calls to the configured Groq API (4 scenarios: simple, multi-part, "
        "multi-source, insufficient-evidence) to confirm the structured-output integration "
        "works end to end with a real key. Never run this automatically - run it by hand "
        "after adding GROQ_API_KEY and setting LLM_ENABLED=True."
    )

    def handle(self, *args, **options):
        try:
            provider = get_groq_provider()
        except GroqConfigurationError as exc:
            raise CommandError(
                f"Groq is not configured, so there is nothing to test: {exc}\n"
                "Set LLM_ENABLED=True and GROQ_API_KEY in your .env first."
            )

        self.stdout.write(f"Using Groq {provider.base_url}, model={provider.model}\n")

        any_failed = False
        for label, question, sources in SCENARIOS:
            self.stdout.write(self.style.MIGRATE_HEADING(label))
            self.stdout.write(f"Question: {question}")
            try:
                result = provider.generate(question, sources)
            except GroqProviderError as exc:
                self.stdout.write(self.style.ERROR(f"  FAILED: {exc}"))
                any_failed = True
                self.stdout.write("")
                continue

            self.stdout.write(self.style.SUCCESS(f"  status: {result.status}"))
            if not result.claims:
                self.stdout.write("  (no claims)")
            for i, claim in enumerate(result.claims, start=1):
                self.stdout.write(f"  claim {i}: {_console_safe(claim.text)}")
                for source in claim.sources:
                    self.stdout.write(
                        f"    - [{source['source_id']}] quote: {_console_safe(source['supporting_quote'])}"
                    )
            self.stdout.write("")

        if any_failed:
            raise CommandError("One or more live Groq scenarios failed - see output above.")
        self.stdout.write(self.style.SUCCESS("All live Groq scenarios completed."))
