"""Illustrative 2-D probing figure: is 'voicing' linearly readable from embeddings?"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(3)

# Two phoneme classes in a 2-D projection of the 768-d embedding space.
# voiceless (t, k, s, p, f...) vs voiced (b, d, z, g, v...), with realistic overlap.
n = 45
voiceless = rng.normal([-1.15, -0.35], [0.85, 0.85], size=(n, 2))
voiced    = rng.normal([1.15, 0.35],  [0.85, 0.85], size=(n, 2))
X = np.vstack([voiceless, voiced])
y = np.array([0] * n + [1] * n)          # 0 = voiceless, 1 = voiced

probe = LogisticRegression().fit(X, y)

# decision-boundary + probability field
xx, yy = np.meshgrid(np.linspace(-3.6, 3.6, 300), np.linspace(-3.2, 3.2, 300))
P = probe.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

C_VL, C_VD = "#E69F00", "#0072B2"   # Okabe-Ito orange / blue
fig, ax = plt.subplots(figsize=(7.2, 5.4))

# faint probability shading
ax.contourf(xx, yy, P, levels=np.linspace(0, 1, 21), cmap="coolwarm", alpha=0.18)
# the decision boundary (P = 0.5)
cs = ax.contour(xx, yy, P, levels=[0.5], colors="#333333", linewidths=2.2, linestyles="--")
ax.clabel(cs, fmt={0.5: "decision boundary"}, fontsize=9)

ax.scatter(*voiceless.T, c=C_VL, s=55, edgecolor="white", linewidth=0.8,
           label="voiceless  (t, k, s, p, f)", zorder=3)
ax.scatter(*voiced.T, c=C_VD, s=55, edgecolor="white", linewidth=0.8,
           label="voiced  (b, d, z, g, v)", zorder=3)

# annotate a few example phonemes near cluster edges
for (px, py), t in [((-1.9, -1.2), "t"), ((-0.5, 0.4), "s"), ((1.9, 1.1), "d"), ((0.4, -0.5), "z")]:
    ax.annotate(t, (px, py), fontsize=13, fontstyle="italic", fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec="#999", lw=0.8))

ax.set_xlabel("embedding — projected dimension 1", fontsize=11)
ax.set_ylabel("embedding — projected dimension 2", fontsize=11)
ax.set_title("Probing 'voicing': can a linear boundary separate the classes?\n"
             "(each dot = one phoneme's 768-d embedding, projected to 2-D)", fontsize=12)
ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
ax.set_xlim(-3.6, 3.6); ax.set_ylim(-3.2, 3.2)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

acc = probe.score(X, y)
ax.text(0.98, 0.03, f"linear probe accuracy = {acc:.0%}", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=10, color="#333",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#ccc"))

fig.tight_layout()
out = "/scratch/claude-215109/-home-woody-vlbi-vlbi108v-BIMAP-FEATURE/5bc33ca5-cba9-41a8-b0f8-9da6fd4f66e2/scratchpad/probe_illustration.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
