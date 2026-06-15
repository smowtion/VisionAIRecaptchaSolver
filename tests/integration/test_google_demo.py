"""Integration test: solve Google's public reCAPTCHA demo.

Opt-in via `pytest -m integration`. Excluded from default pytest run because it
launches a real Chromium browser, pulls model weights on first use, and depends
on Google's demo endpoint staying reachable.

reCAPTCHA serves challenges non-deterministically (3x3 / 4x4, varying classes), so a
single solve can land on a challenge that genuinely cannot be solved this round. The
test retries `solve` a few times within a wall-clock budget (cheap now that unsupported
challenges fast-skip) and still asserts a real token — the assertion is not relaxed,
only made resilient to non-determinism.
"""

from __future__ import annotations

import time

import pytest

from vision_ai_recaptcha_solver import RecaptchaSolver, SolverConfig, TokenExtractionError

GOOGLE_DEMO_SITEKEY = "6Le-wvkSAAAAAPBMRTvw0Q4Muexq9bi0DJwx_mJ-"
GOOGLE_DEMO_URL = "https://www.google.com/recaptcha/api2/demo"

MAX_SOLVE_RETRIES = 3
WALL_CLOCK_BUDGET_SECONDS = 180.0


@pytest.mark.integration
def test_solve_google_demo_headless() -> None:
    """Solve Google's reCAPTCHA v2 demo (retry-until-solvable) and verify the token."""
    config = SolverConfig(
        headless=True,
        timeout=120.0,
        log_level="WARNING",
    )

    deadline = time.monotonic() + WALL_CLOCK_BUDGET_SECONDS
    result = None
    last_error: Exception | None = None

    with RecaptchaSolver(config) as solver:
        for _attempt in range(MAX_SOLVE_RETRIES):
            if time.monotonic() >= deadline:
                break
            try:
                result = solver.solve(
                    website_key=GOOGLE_DEMO_SITEKEY,
                    website_url=GOOGLE_DEMO_URL,
                )
                break
            except TokenExtractionError as e:
                # Non-deterministic unsolvable challenge this round; try a fresh solve.
                last_error = e
                continue

    assert result is not None, (
        f"all {MAX_SOLVE_RETRIES} solve attempts failed within "
        f"{WALL_CLOCK_BUDGET_SECONDS:.0f}s (last error: {last_error})"
    )
    assert result.token, "solve returned empty token"
    assert isinstance(result.token, str)
    assert result.time_taken > 0
    assert result.attempts >= 1
    assert result.captcha_type.value != "unknown"
