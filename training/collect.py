"""Drive repeated solves with collection enabled to accumulate training data.

Runs the solver N times against a reCAPTCHA page with ``collect_data=True``, so the
DataCollector accumulates per-cell tiles (classification) + full 4x4 images (detection)
under ``collect_dir``. Each solve is best-effort (failures are counted, not fatal). A delay
between runs avoids hammering the target.

Usage::

    python training/collect.py --runs 200 --delay 3 --collect-dir collected
"""

from __future__ import annotations

import time
from pathlib import Path

import click


def count_full_images(collect_dir: Path) -> int:
    """Count collected full 4x4 detection images under ``<collect_dir>/full/``."""
    full = Path(collect_dir) / "full"
    if not full.exists():
        return 0
    return sum(1 for _ in full.rglob("*.png"))


def count_tiles(collect_dir: Path) -> int:
    """Count collected per-cell classification tiles (excludes the full/ subdir)."""
    root = Path(collect_dir)
    if not root.exists():
        return 0
    full = root / "full"
    return sum(1 for p in root.rglob("*.png") if full not in p.parents)


@click.command()
@click.option("--runs", default=100, type=int, help="Number of solve attempts.")
@click.option("--delay", default=2.0, type=float, help="Seconds between runs (be polite).")
@click.option("--collect-dir", default="collected", help="Collection output directory.")
@click.option(
    "--site-key",
    default="6Le-wvkSAAAAAPBMRTvw0Q4Muexq9bi0DJwx_mJ-",
    help="reCAPTCHA site key (default: Google demo).",
)
@click.option(
    "--url",
    default="https://www.google.com/recaptcha/api2/demo",
    help="Page URL (default: Google demo).",
)
@click.option("--headless/--headed", default=True, help="Run browser headless.")
def main(
    runs: int, delay: float, collect_dir: str, site_key: str, url: str, headless: bool
) -> None:
    """Loop solves with collection on; report progress + collected counts."""
    # Deferred import so --help works without the full runtime installed.
    from vision_ai_recaptcha_solver import RecaptchaSolver, SolverConfig

    config = SolverConfig(
        collect_data=True,
        collect_dir=collect_dir,
        headless=headless,
        log_level="ERROR",
    )

    solved = 0
    failed = 0
    for i in range(1, runs + 1):
        try:
            with RecaptchaSolver(config) as solver:
                solver.solve(website_key=site_key, website_url=url)
            solved += 1
        except Exception as e:  # best-effort data collection; keep going
            failed += 1
            click.echo(f"  run {i}: {type(e).__name__}: {e}")

        full = count_full_images(Path(collect_dir))
        tiles = count_tiles(Path(collect_dir))
        click.echo(
            f"[{i}/{runs}] solved={solved} failed={failed} "
            f"| collected: {full} full-4x4, {tiles} tiles"
        )
        if i < runs:
            time.sleep(delay)

    click.echo(
        f"\nDone. {solved}/{runs} solved. Collected {count_full_images(Path(collect_dir))} "
        f"full 4x4 images, {count_tiles(Path(collect_dir))} tiles in {collect_dir}/"
    )


if __name__ == "__main__":
    main()
