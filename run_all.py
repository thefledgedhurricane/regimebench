"""Reproduce every artifact the manuscript cites, in dependency order.

    python run_all.py              # everything
    python run_all.py --list       # show the pipeline without running it
    python run_all.py --from 6     # resume at step 6
    python run_all.py --only 1 4   # run selected steps

Steps run as subprocesses with UTF-8 forced, so the pipeline behaves identically on a
Windows console (cp1252 by default) and on POSIX. Any non-zero exit stops the run and
propagates the code. A step whose script is not present is skipped rather than failing.
"""

import argparse
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))

PIPELINE = [
    ("Factor grid",                      "experiments/run_pilot_grid.py"),
    ("Positive control (harness)",       "experiments/run_positive_control.py"),
    ("Positive control (criterion)",     "experiments/run_criterion_control.py"),
    ("Mechanism ablation",               "experiments/run_mechanism_ablation.py"),
    ("Separation sweep",                 "experiments/run_separation_sweep.py"),
    ("Null controls (3 processes)",      "experiments/run_three_null_controls.py"),
    ("SV+jump null control",             "experiments/run_sv_jump_robustness.py"),
    ("Gap reference study",              "experiments/run_gap_reference_study.py"),
    ("Gap curve diagnostic",             "experiments/run_gap_diagnostic.py"),
    ("Feature-representation grid",      "experiments/run_feature_grid.py"),
    ("4-D multivariate test",            "experiments/run_multivariate_test.py"),
    ("Binance high-frequency check",     "experiments/run_binance_benchmark.py"),
    ("Real-data agreement (1926-2026)",  "experiments/run_real_data_agreement.py"),
    ("Permutation test (both refs)",     "experiments/run_permutation_test.py"),
    ("Figures (png + vector pdf)",       "experiments/generate_paper_figures.py"),
    ("Graphical abstract",               "experiments/generate_graphical_abstract.py"),
    ("Manuscript integrity verifier",    "tools/verify_manuscript.py"),
]


def run_step(index: int, name: str, script: str) -> float:
    print("\n" + "=" * 90)
    print(f"STEP {index}/{len(PIPELINE)}: {name}  ({script})")
    print("=" * 90, flush=True)

    if not os.path.isfile(os.path.join(REPO, script)):
        print(f"SKIPPED: {script} is not present in this checkout", flush=True)
        return 0.0

    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    t0 = time.time()
    code = subprocess.call([sys.executable, "-X", "utf8", "-u", os.path.join(REPO, script)],
                           cwd=REPO, env=env)
    elapsed = time.time() - t0

    if code != 0:
        print(f"\nFAILED: {name} exited with code {code} after {elapsed:.1f}s")
        sys.exit(code)
    print(f"\nOK: {name} completed in {elapsed:.1f}s", flush=True)
    return elapsed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the pipeline and exit")
    ap.add_argument("--from", dest="start", type=int, default=1, help="resume at this step")
    ap.add_argument("--only", nargs="+", type=int, help="run only these step numbers")
    args = ap.parse_args()

    if args.list:
        for i, (name, script) in enumerate(PIPELINE, 1):
            print(f"  {i:2d}. {name:34s} {script}")
        return

    selected = [(i, n, s) for i, (n, s) in enumerate(PIPELINE, 1)
                if (args.only and i in args.only) or (not args.only and i >= args.start)]

    print("=" * 90)
    print(f"REGIMEBENCH REPRODUCIBILITY RUNNER — {len(selected)} of {len(PIPELINE)} steps")
    print("=" * 90)

    t0 = time.time()
    timings = [(name, run_step(i, name, script)) for i, name, script in selected]

    print("\n" + "=" * 90)
    print(f"PIPELINE COMPLETE in {time.time() - t0:.1f}s")
    for name, el in sorted(timings, key=lambda x: -x[1]):
        print(f"  {el:8.1f}s  {name}")
    print("=" * 90)


if __name__ == "__main__":
    main()
