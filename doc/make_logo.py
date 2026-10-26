"""
metalc logo: "LC" in Canela Text, rendered as a sparse Gaussian point cloud.

Each anchor point is rejection-sampled from the glyph silhouette (weighted by
alpha), then spawns a small cluster of datapoints drawn from a 2D Gaussian
centered on that anchor -- so the letterforms emerge from the point cloud
rather than being filled solid.

Regenerate with: uv run python logo/make_logo.py
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

FONT_PATH = "/Users/huopeiyang/Library/CloudStorage/OneDrive-KULeuven/py_working/EuroVIS_2026/poster Folder/Document fonts/CanelaText.ttc"
OUT_DIR = "/Users/huopeiyang/Library/CloudStorage/OneDrive-KULeuven/py_working/semi-LDA/logo"

W, H = 2400, 1000
FONT_SIZE = 620
COLOR = "#121212"
SEED = 674


def rasterize_mask(text, font_index):
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE, index=font_index)
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pos = ((W - tw) / 2 - bbox[0], (H - th) / 2 - bbox[1])
    draw.text(pos, text, font=font, fill=255)
    return np.array(img)


mask = rasterize_mask("LC", font_index=2)  # Regular
alpha = mask.astype(float) / 255.0


def make_points(seed):
    rng = np.random.default_rng(seed)

    n_candidates = 400_000
    xs = rng.uniform(0, W, n_candidates)
    ys = rng.uniform(0, H, n_candidates)
    xi, yi = xs.astype(int).clip(0, W - 1), ys.astype(int).clip(0, H - 1)
    p = alpha[yi, xi]
    keep = rng.uniform(0, 1, n_candidates) < p
    ax_, ay_ = xs[keep], ys[keep]

    n_anchors = 220
    idx = rng.choice(len(ax_), n_anchors, replace=False)
    ax_, ay_ = ax_[idx], ay_[idx]

    sigma = 20.0
    cluster_size = 4
    n_points = n_anchors * cluster_size
    gx = rng.normal(0, sigma, n_points) + np.repeat(ax_, cluster_size)
    gy = rng.normal(0, sigma, n_points) + np.repeat(ay_, cluster_size)

    gxi, gyi = gx.astype(int).clip(0, W - 1), gy.astype(int).clip(0, H - 1)
    local_a = alpha[gyi, gxi]
    soft_keep = rng.uniform(0, 1, n_points) < np.clip(local_a + 0.3, 0, 1)
    gx, gy = gx[soft_keep], gy[soft_keep]

    sizes = rng.uniform(5, 30, len(gx)) ** 1.5
    alphas = rng.uniform(0.35, 0.95, len(gx))
    return gx, gy, sizes, alphas


gx, gy, sizes, alphas = make_points(seed=SEED)

fig, ax = plt.subplots(figsize=(W / 200, H / 200), dpi=200)
ax.scatter(gx, H - gy, s=sizes, c=COLOR, alpha=alphas, linewidths=0, edgecolors="none")
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)
plt.tight_layout(pad=0)

out_png = f"{OUT_DIR}/metalc_logo.png"
fig.savefig(out_png, transparent=True, dpi=200)
plt.close(fig)

im = Image.open(out_png).convert("RGBA")
bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
bg.alpha_composite(im)
bg.convert("RGB").save(f"{OUT_DIR}/metalc_logo_onwhite.png")

print("saved", len(gx), "points")
