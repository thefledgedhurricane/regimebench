"""Build the Elsevier submission package.

Elsevier's editorial system wants figures as separate uploads with logical names
(fig1.pdf, fig2.pdf, ...), not as relative paths reaching outside the manuscript
directory. This script produces that package without touching the source manuscript:

    manuscript/manuscript_array.tex   stays the source of truth, keeps its
                                      ../experiments/output/ paths, and keeps
                                      compiling and verifying in place.

    manuscript/submission/            self-contained upload set:
                                        manuscript.tex   (figN references)
                                        fig1.pdf ... figN.pdf
                                        highlights.txt

Figures are numbered by order of first appearance in the manuscript, which is the
order Elsevier expects and the order the captions already imply.

    python tools/package_submission.py
    python tools/package_submission.py --check   # verify only, write nothing
"""

import argparse
import os
import re
import shutil
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEX = os.path.join(REPO, "manuscript", "manuscript_array.tex")
FIGDIR = os.path.join(REPO, "experiments", "output")
OUTDIR = os.path.join(REPO, "manuscript", "submission")
HIGHLIGHTS = os.path.join(REPO, "manuscript", "highlights.txt")

INCLUDE = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?\{)([^}]*?)(\})")


def figure_order(tex: str):
    """Stems of the figures, in order of first appearance, deduplicated."""
    order, seen = [], set()
    for m in INCLUDE.finditer(tex):
        stem = os.path.basename(m.group(2))
        if stem not in seen:
            seen.add(stem)
            order.append(stem)
    return order


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report what would be written, change nothing")
    args = ap.parse_args()

    with open(TEX, encoding="utf-8") as f:
        tex = f.read()

    order = figure_order(tex)
    if not order:
        print("ERROR: no \\includegraphics found in the manuscript", file=sys.stderr)
        return 1

    mapping = {stem: f"fig{i}" for i, stem in enumerate(order, 1)}

    # Every figure must exist as vector PDF. Elsevier asks 500 dpi for combination
    # artwork and 1000 dpi for line art; vector sidesteps the raster requirement.
    missing = [s for s in order if not os.path.isfile(os.path.join(FIGDIR, s + ".pdf"))]
    if missing:
        print("ERROR: missing vector PDF for: " + ", ".join(missing), file=sys.stderr)
        print("Run: python experiments/generate_paper_figures.py", file=sys.stderr)
        return 1

    packaged = INCLUDE.sub(
        lambda m: m.group(1) + mapping[os.path.basename(m.group(2))] + m.group(3), tex)

    if INCLUDE.search(packaged) and "../experiments/output" in packaged:
        print("ERROR: a relative figure path survived rewriting", file=sys.stderr)
        return 1

    for stem, name in mapping.items():
        print(f"  {name}.pdf  <- {stem}.pdf")

    if args.check:
        print(f"\n--check: {len(mapping)} figures resolve, nothing written.")
        return 0

    os.makedirs(OUTDIR, exist_ok=True)
    for stem, name in mapping.items():
        shutil.copy2(os.path.join(FIGDIR, stem + ".pdf"),
                     os.path.join(OUTDIR, name + ".pdf"))

    with open(os.path.join(OUTDIR, "manuscript.tex"), "w", encoding="utf-8") as f:
        f.write(packaged)

    if os.path.isfile(HIGHLIGHTS):
        shutil.copy2(HIGHLIGHTS, os.path.join(OUTDIR, "highlights.txt"))

    # The graphical abstract is a separate upload in Elsevier's system, not a figure in
    # the article, so it is copied under its own name and never enters the figN mapping.
    extras = []
    for ext in ("pdf", "png"):
        src = os.path.join(FIGDIR, f"graphical_abstract.{ext}")
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(OUTDIR, f"graphical_abstract.{ext}"))
            extras.append(f"graphical_abstract.{ext}")
    if not extras:
        print("  note: no graphical abstract found -- "
              "run python experiments/generate_graphical_abstract.py")

    print(f"\nWrote {len(mapping)} figures + manuscript.tex to manuscript/submission/")
    if extras:
        print("  plus " + ", ".join(extras) + " (uploaded separately, not an article figure)")
    print("Compile there to confirm the package is self-contained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
