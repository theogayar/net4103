import argparse
import os
import subprocess
import sys
import time

SCRIPTS = [
    "question2.py",
    "question3.py",
    "question4.py",
    "question4e.py",
    "question5.py",
    "question6.py",
]


def main():
    parser = argparse.ArgumentParser(description="Run every question script in order.")
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="NAME",
        help="run only these scripts, e.g. --only question2 question6",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        metavar="NAME",
        help="skip these scripts, e.g. --skip question4e",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue with the next script when one fails",
    )
    args = parser.parse_args()

    def stem(name):
        return name.removesuffix(".py")

    selected = [
        s for s in SCRIPTS if args.only is None or stem(s) in map(stem, args.only)
    ]
    selected = [s for s in selected if stem(s) not in map(stem, args.skip)]
    if not selected:
        parser.error(f"no scripts selected; available: {', '.join(map(stem, SCRIPTS))}")

    # Scripts are run from the repository root so their relative "data/" and "report/" paths resolve.
    root = os.path.dirname(os.path.abspath(__file__))
    # Non-interactive backend: figures are still saved, but plt.show() no longer blocks.
    env = {**os.environ, "MPLBACKEND": "Agg"}

    summary = []
    for script in selected:
        print(f"\n===== {script} =====", flush=True)
        start = time.perf_counter()
        returncode = subprocess.run(
            [sys.executable, script], check=False, cwd=root, env=env
        ).returncode
        elapsed = time.perf_counter() - start
        summary.append((script, returncode, elapsed))
        if returncode != 0 and not args.keep_going:
            break

    print("\n===== Summary =====")
    for script, returncode, elapsed in summary:
        status = "ok" if returncode == 0 else f"FAILED (exit {returncode})"
        print(f" {script:14} {status:20} {elapsed:8.1f}s")
    for script in selected[len(summary) :]:
        print(f" {script:14} not run")

    sys.exit(
        0
        if all(rc == 0 for _, rc, _ in summary) and len(summary) == len(selected)
        else 1
    )


if __name__ == "__main__":
    main()
