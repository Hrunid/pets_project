"""Run the experiment, generate its report, and audit the saved results."""
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
STEPS = ("experiment.py", "make_report.py", "audit_results.py")


def main() -> int:
    for script in STEPS:
        print(f"\n=== Running {script} ===", flush=True)
        result = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
        if result.returncode:
            print(
                f"{script} failed with exit code {result.returncode}; "
                "remaining steps were not run.",
                file=sys.stderr,
            )
            return result.returncode

    print("\nExperiment, report, and audit completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
