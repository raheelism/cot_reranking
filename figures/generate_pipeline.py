"""
Generate fig_pipeline.png — high-visibility system pipeline diagram
"When Does Chain-of-Thought Hurt Reranking?"
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Colour palette ─────────────────────────────────────────────────────────
C_BLUE   = "#1F5C8B"
C_TEAL   = "#1A7A6E"
C_AMBER  = "#B05A10"
C_PURPLE = "#5B3A8C"
C_GREY   = "#333333"
C_BG     = "#F8F9FA"

FILL_BLUE   = "#E3EFF8"
FILL_TEAL   = "#DFF5F1"
FILL_AMBER  = "#FFF3E3"
FILL_PURPLE = "#EDE8F6"
FILL_LGREY  = "#EEF1F4"

# ── Figure ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 10))
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)
ax.set_xlim(0, 18.4)
ax.set_ylim(0, 8)
ax.axis('off')

# ── Helpers ────────────────────────────────────────────────────────────────

def box(cx, cy, w, h, fc, ec, lw=2.2, r=0.28, z=3):
    p = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z
    )
    ax.add_patch(p)

def arr(x1, y1, x2, y2, color, lw=2.2, z=2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=22,
                                connectionstyle="arc3,rad=0"),
                zorder=z)

def txt(x, y, s, size=11, color=C_GREY, weight='normal',
        ha='center', va='center', z=5, style='normal'):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=z, style=style,
            fontfamily='DejaVu Sans')

# ── Stage header bands ─────────────────────────────────────────────────────
stage_data = [
    (0.05, 2.75,  "Stage 1",   "Retrieval",  C_BLUE),
    (2.75, 9.05,  "Stage 2",   "Reranking",  C_GREY),
    (9.05, 17.95, "Stage 3",   "Analysis",   C_PURPLE),
]
for x0, x1, s1, s2, col in stage_data:
    ax.add_patch(plt.Rectangle((x0, 7.25), x1 - x0, 0.65,
                               facecolor=col, alpha=0.08, zorder=1,
                               linewidth=0))
    txt((x0 + x1) / 2, 7.57, s1, 20, col, 'bold', style='italic')
    txt((x0 + x1) / 2, 7.28, s2, 18, col, style='italic')

# vertical separators
for xs in [2.75, 9.05]:
    ax.plot([xs, xs], [0.3, 7.25], color='#BBBBBB', lw=1.5, ls='--', zorder=1)

# ── STAGE 1: Query + BM25 + Top-100 ───────────────────────────────────────
# Query
box(0.95, 4.0, 1.55, 1.35, FILL_LGREY, C_BLUE, lw=2.5)
txt(0.95, 4.28, "Query", 22, C_BLUE, 'bold')
txt(0.95, 3.78, r"$q$", 20, C_GREY)

# BM25
box(2.10, 4.0, 1.55, 1.35, FILL_BLUE, C_BLUE, lw=2.5)
txt(2.10, 4.30, "BM25", 22, C_BLUE, 'bold')
txt(2.10, 3.85, "First-stage", 15, C_GREY)
txt(2.10, 3.55, "retrieval", 15, C_GREY)

# Top-100
box(3.55, 4.0, 1.60, 1.35, FILL_BLUE, C_BLUE, lw=2.5)
txt(3.55, 4.32, "Top-100", 22, C_BLUE, 'bold')
txt(3.55, 3.88, r"Candidates", 15, C_GREY)
txt(3.55, 3.60, r"$\mathcal{D}_q$", 17, C_GREY)

# arrows stage 1
arr(1.70, 4.0, 1.32, 4.0, C_BLUE)
arr(2.87, 4.0, 2.70, 4.0, C_BLUE)
arr(3.24, 4.0, 2.88, 4.0, C_BLUE)

# ── Fork from Top-100 ─────────────────────────────────────────────────────
fork_x = 4.90
ax.plot([4.36, fork_x], [4.0, 4.0], color=C_GREY, lw=2.2, zorder=2)
ax.plot([fork_x, fork_x], [2.20, 5.80], color=C_GREY, lw=2.2, zorder=2)

# ── STAGE 2: Direct-Point branch (top) ───────────────────────────────────
dp_y = 5.75
arr(fork_x, dp_y, 5.55, dp_y, C_TEAL, lw=2.5)
box(7.00, dp_y, 2.80, 1.60, FILL_TEAL, C_TEAL, lw=2.6)
txt(7.00, dp_y + 0.42, "Direct-Point", 22, C_TEAL, 'bold')
txt(7.00, dp_y - 0.05, "Logit at final input token", 15, C_GREY)
txt(7.00, dp_y - 0.44, r"No generation required", 13, C_GREY, style='italic')

# ── STAGE 2: Reason-Point branch (bottom) ─────────────────────────────────
rp_y = 2.25
arr(fork_x, rp_y, 5.55, rp_y, C_AMBER, lw=2.5)
box(7.00, rp_y, 2.80, 1.60, FILL_AMBER, C_AMBER, lw=2.6)
txt(7.00, rp_y + 0.42, "Reason-Point", 22, C_AMBER, 'bold')
txt(7.00, rp_y - 0.05, r"CoT trace $\rightarrow$ decision logit", 15, C_GREY)
txt(7.00, rp_y - 0.44, r"$\leq$128 generated tokens", 13, C_AMBER, style='italic')

# ── Merge and score ────────────────────────────────────────────────────────
merge_x = 8.60
arr(8.41, dp_y, merge_x, dp_y, C_TEAL, lw=2.5)
arr(8.41, rp_y, merge_x, rp_y, C_AMBER, lw=2.5)
ax.plot([merge_x, merge_x], [rp_y, dp_y], color=C_GREY, lw=2.5, zorder=2)
ax.plot([merge_x, 9.20], [4.0, 4.0], color=C_GREY, lw=2.5, zorder=2)
txt(8.90, 4.35, r"$s(q,d)$", 18, C_GREY)

# ── NDCG@10 box ───────────────────────────────────────────────────────────
ndcg_x = 10.35
arr(9.20, 4.0, 9.72, 4.0, C_PURPLE, lw=2.5)
box(ndcg_x, 4.0, 2.30, 1.55, FILL_PURPLE, C_PURPLE, lw=2.6)
txt(ndcg_x, 4.35, "NDCG@10", 22, C_PURPLE, 'bold')
txt(ndcg_x, 3.88, "Per-query scores", 15, C_GREY)
txt(ndcg_x, 3.58, "5 datasets", 13, C_GREY, style='italic')

# ── Three analysis modules ─────────────────────────────────────────────────
sub_x_line = 11.72
arr(11.50, 4.0, sub_x_line, 4.0, C_PURPLE, lw=2.5)
ax.plot([sub_x_line, sub_x_line], [1.90, 6.10],
        color=C_PURPLE, lw=2.0, ls='dotted', zorder=2)

modules = [
    (5.80, "#1A7A6E",  "Complexity\nStratification",
     "RQ2: CoT effect\nby query tercile"),
    (4.00, "#5B3A8C",  "CoT Length\nAnalysis",
     r"RQ3: Pearson $r$, Spearman $\rho$"),
    (2.20, "#B05A10",  "Selective\nRouting",
     "RQ4: Direct vs. Router"),
]

fills_m = [FILL_TEAL, FILL_PURPLE, FILL_AMBER]
edges_m = [C_TEAL, C_PURPLE, C_AMBER]

for (my, ec, title, sub), fc in zip(modules, fills_m):
    arr(sub_x_line, my, 12.12, my, ec, lw=2.2)
    box(14.95, my, 5.40, 1.62, fc, ec, lw=2.4, r=0.32)
    lines = title.split('\n')
    if len(lines) == 2:
        txt(14.95, my + 0.38, lines[0], 20, ec, 'bold')
        txt(14.95, my + 0.02, lines[1], 20, ec, 'bold')
    else:
        txt(14.95, my + 0.18, lines[0], 20, ec, 'bold')
    txt(14.95, my - 0.45, sub, 14, C_GREY)

# ── Scoring formula box ────────────────────────────────────────────────────
ax.text(7.00, 0.82,
        r"$s(q,d)=\mathrm{softmax}\!\left([\ell_{\mathrm{true}},\,"
        r"\ell_{\mathrm{false}}]\right)[0]$",
        fontsize=19, color=C_GREY, ha='center', va='center', zorder=6,
        bbox=dict(facecolor='white', edgecolor='#AAAAAA',
                  boxstyle='round,pad=0.7', linewidth=2.0, alpha=0.95))

# ── Legend ────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor=FILL_TEAL,   edgecolor=C_TEAL,
                   linewidth=1.8, label="Direct-Point path"),
    mpatches.Patch(facecolor=FILL_AMBER,  edgecolor=C_AMBER,
                   linewidth=1.8, label="Reason-Point path"),
    mpatches.Patch(facecolor=FILL_PURPLE, edgecolor=C_PURPLE,
                   linewidth=1.8, label="Analysis module"),
]
leg = ax.legend(handles=legend_patches, loc='lower right',
                fontsize=16, framealpha=0.95, edgecolor='#AAAAAA',
                handlelength=1.8, handleheight=1.6,
                bbox_to_anchor=(0.998, 0.02))

plt.tight_layout(pad=0.5)
plt.savefig("fig_pipeline.png", dpi=220, bbox_inches='tight',
            facecolor=C_BG, edgecolor='none')
print("Saved fig_pipeline.png")
