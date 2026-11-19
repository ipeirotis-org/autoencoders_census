"""
Standardize Dataset Naming Across Figures 4 and 8 (Referee Minor 3.2):
  1. Regenerate figures/reconstruction_ora.pdf from the verified 27 ORA values in
     Table tab:reconstruction_perf (appendices.tex) with full Author (Year) labels.
  2. Losslessly update panel titles in figures/fairness_frontier.pdf:
       'alvarez2019  (well-powered)'                -> 'Alvarez et al. (2019)  (well-powered)'
       'robinson2014  (well-powered)'               -> 'Kann et al. (2018)  (well-powered)'
       'ogrady2019  (gender axis only; indicative)' -> "O'Grady (2019)  (gender axis; indicative)"
"""
import os
import shutil
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.environ.get("PAPER_DIR", os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "inattentiveness_paper")))
FIG_DIR = os.environ.get("FIG_DIR", os.path.join(PAPER_DIR, "figures"))
BACKUP_DIR = os.environ.get("BACKUP_DIR", "")
if BACKUP_DIR:
    os.makedirs(BACKUP_DIR, exist_ok=True)

TTF_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
DEJAVU_SERIF = os.path.join(TTF_DIR, "DejaVuSerif.ttf")


def backup_once(fname):
    if not BACKUP_DIR:
        return
    src = os.path.join(FIG_DIR, fname)
    dst = os.path.join(BACKUP_DIR, fname)
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)


def regenerate_reconstruction_ora():
    backup_once("reconstruction_ora.pdf")
    rows = [
        ("Pennycook et al. (2020)",      0.79, 0.77, 0.70),
        ("Kann et al. (2018)",           0.75, 0.73, 0.79),
        ("Ivanov et al. (2021)",         0.81, 0.78, 0.81),
        ("O'Grady (2019)",               0.84, 0.77, 0.83),
        ("Mastroianni & Dana (2022)",    0.83, 0.81, 0.86),
        ("Uhalt (2020)",                 0.87, 0.82, 0.85),
        ("Alvarez et al. (2019)",        0.88, 0.86, 0.93),
        ("Buchanan & Scofield (2018)",   0.93, 0.82, 0.97),
        ("Moss et al. (2023)",           0.94, 0.88, 0.98),
    ]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10.5,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 9.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(7.2, 4.05))
    ys = np.arange(len(rows))

    for y_i, (_, p100, p85, lin) in enumerate(rows):
        lo = min(p100, p85, lin)
        hi = max(p100, p85, lin)
        ax.plot([lo, hi], [y_i, y_i], color="#CCCCCC", linewidth=2.4, zorder=2)

    p100_vals = [r[1] for r in rows]
    p85_vals = [r[2] for r in rows]
    lin_vals = [r[3] for r in rows]

    ax.scatter(p100_vals, ys, s=62, color="#0072B2", label=r"AE ($p = 100$)", zorder=3)
    ax.scatter(p85_vals, ys, s=62, color="#E69F00", label=r"AE + PL ($p = 85$)", zorder=4)
    ax.scatter(lin_vals, ys, s=62, color="#009E73", label="Linear AE", zorder=5)

    ax.axvline(0.50, color="#777777", linestyle="--", linewidth=1.2, zorder=1)
    ax.text(0.506, len(rows) - 0.82, "chance (0.5)", color="#666666", fontsize=9.5, va="center")

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0.48, 1.01)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylim(-0.6, len(rows) - 0.35)
    ax.set_xlabel("Reconstruction one-vs-all ROC AUC")
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "reconstruction_ora.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("Regenerated", out_path)


def replace_titles_in_pdf(fname, replacements, fontfile, fontname):
    try:
        import pymupdf
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required to relabel PDF panel titles; install via "
            "`pip install -r scripts/misq_revision/requirements-revision.txt`."
        ) from exc

    backup_once(fname)
    dst = os.path.join(FIG_DIR, fname)
    src = os.path.join(BACKUP_DIR, fname) if BACKUP_DIR else dst
    if not os.path.exists(src):
        raise FileNotFoundError(f"Input figure PDF not found: {src}")

    doc = pymupdf.open(src)
    page = doc[0]
    font_obj = pymupdf.Font(fontfile=fontfile)

    matched = 0
    spans_to_edit = []
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for s in line.get("spans", []):
                txt = s["text"]
                for old_sub, new_txt in replacements.items():
                    if old_sub in txt:
                        spans_to_edit.append((s, new_txt))
                        break

    if not spans_to_edit:
        doc.close()
        print(f"No legacy panel titles remaining in {dst} (already standardized)")
        return

    for s, _new_txt in spans_to_edit:
        bbox = pymupdf.Rect(s["bbox"])
        pad_rect = pymupdf.Rect(bbox.x0 - 4, bbox.y0 - 1, bbox.x1 + 4, bbox.y1 + 1)
        page.add_redact_annot(pad_rect, fill=(1, 1, 1))
        matched += 1

    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE)

    page.insert_font(fontname=fontname, fontfile=fontfile)
    for s, new_txt in spans_to_edit:
        bbox = pymupdf.Rect(s["bbox"])
        sz = s["size"]
        orig_y = s["origin"][1]
        center_x = 0.5 * (bbox.x0 + bbox.x1)
        new_w = font_obj.text_length(new_txt, fontsize=sz)
        new_x = center_x - 0.5 * new_w
        page.insert_text(
            pymupdf.Point(new_x, orig_y),
            new_txt,
            fontname=fontname,
            fontsize=sz,
            color=(0, 0, 0),
        )

    tmp_dst = dst + ".tmp"
    doc.save(tmp_dst, garbage=4, deflate=True)
    doc.close()
    os.replace(tmp_dst, dst)
    print(f"Updated {matched} panel titles in {dst}")


if __name__ == "__main__":
    regenerate_reconstruction_ora()
    replace_titles_in_pdf(
        "fairness_frontier.pdf",
        {
            "alvarez2019": "Alvarez et al. (2019)  (well-powered)",
            "robinson2014": "Kann et al. (2018)  (well-powered)",
            "ogrady2019": "O'Grady (2019)  (gender axis; indicative)",
        },
        DEJAVU_SERIF,
        "F_DejaVuSerif",
    )
