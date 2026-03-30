"""
Visualize encoder representations on weather data via PCA and t-SNE.
Checks for representation collapse by plotting patch and semantic embeddings.

Usage:
    python visualize_representations.py --checkpoint ./output_model/DiscreteJEPA/_epoch500best_model.pt
    python visualize_representations.py  # auto-finds latest checkpoint
"""

import argparse
import sys
import os
import glob
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(__file__))

from config_files.config_pretrain import config
from Discrete_JEPA.Encoder import Encoder
from Discrete_JEPA.VQ import VectorQuantizer
from data_loaders.data_puller import ForcastingDataPullerDescrete

# ── helpers ──────────────────────────────────────────────────────────────────

def find_latest_checkpoint(base_dir="./output_model/DiscreteJEPA"):
    """Return path of checkpoint with highest epoch number."""
    pattern = os.path.join(base_dir, "_epoch*best_model.pt")
    files = glob.glob(pattern)
    if not files:
        # fall back to the best model
        fallback = os.path.join(base_dir, "best_model.pt")
        if os.path.exists(fallback):
            return fallback
        raise FileNotFoundError(f"No checkpoint found under {base_dir}")
    # sort by epoch number
    def epoch_num(p):
        try:
            return int(os.path.basename(p).split("_epoch")[1].split("best")[0])
        except Exception:
            return -1
    return max(files, key=epoch_num)


def build_encoder():
    return Encoder(
        num_patches=config["ratio_patches"],
        num_semantic_tokens=config["num_semantic_tokens"],
        dim_in=config["patch_size"],
        embed_dim=config["encoder_embed_dim"],
        nhead=config["nhead"],
        num_layers=config["num_encoder_layers"],
        mlp_ratio=config["mlp_ratio"],
        drop_rate=0.0,
        attn_drop_rate=0.0,
        pe='sincos',
        learn_pe=False,
        res_attention=True,
    )


def build_vq():
    return VectorQuantizer(
        num_embeddings=config["codebook_size"],
        embedding_dim=config["encoder_embed_dim"],
        commitment_cost=config["commitment_cost"],
    )


# ── data collection ──────────────────────────────────────────────────────────

@torch.no_grad()
def collect_embeddings(encoder, vq, loader, device, max_batches=30):
    """
    Returns dicts with collected arrays:
      patch_emb  : [N_patches, embed_dim]   — per-patch embeddings
      sem_emb    : [N_sem, embed_dim]        — per-semantic-token embeddings
      sem_codes  : [N_sem]                   — VQ code index per semantic token
      var_labels : [N_patches]               — variable index (0..n_vars-1)
      pos_labels : [N_patches]               — patch position (0..num_patches-1)
      var_labels_sem : [N_sem]
    """
    encoder.eval()
    vq.eval()

    patch_embs, sem_embs, sem_codes_list = [], [], []
    var_labels, pos_labels, var_labels_sem = [], [], []

    n_collected = 0
    for context_patches, _ in loader:
        if n_collected >= max_batches:
            break

        if context_patches.dim() == 3:
            context_patches = context_patches.unsqueeze(-1)
        context_patches = context_patches.to(device)  # [B, T, P_L, n_v]
        B, T, P_L, n_v = context_patches.shape

        out = encoder(context_patches)
        data_patches = out["data_patches"]           # [B*n_v, T, D]
        quantized_sem = out["quantized_semantic"]    # [B*n_v, S, D]

        # VQ encode semantic tokens → returns (loss, quantized, perplexity, encoding_indices, encodings, soft_avg)
        _, _, _, encoding_indices, _, _ = vq(quantized_sem)
        # encoding_indices: [(B*n_v*S), 1]
        S = quantized_sem.shape[1]
        code_idx = encoding_indices.reshape(B * n_v, S)  # [B*n_v, S]

        # Patch embeddings: [B*n_v, T, D] → flatten first two dims
        patch_embs.append(data_patches.reshape(-1, config["encoder_embed_dim"]).cpu().numpy())

        # Variable label for each patch token
        var_idx = torch.arange(n_v, device=device).repeat_interleave(1).repeat(B)  # [B*n_v]
        # expand to [B*n_v, T]
        var_idx_patch = var_idx.unsqueeze(1).expand(B * n_v, T).reshape(-1)
        var_labels.append(var_idx_patch.cpu().numpy())

        # Patch position label [0..T-1]
        pos_idx = torch.arange(T, device=device).unsqueeze(0).expand(B * n_v, T).reshape(-1)
        pos_labels.append(pos_idx.cpu().numpy())

        # Semantic embeddings
        S = quantized_sem.shape[1]
        sem_embs.append(quantized_sem.reshape(-1, config["encoder_embed_dim"]).cpu().numpy())
        sem_codes_list.append(code_idx.reshape(-1).cpu().numpy())

        var_idx_sem = var_idx.unsqueeze(1).expand(B * n_v, S).reshape(-1)
        var_labels_sem.append(var_idx_sem.cpu().numpy())

        n_collected += 1

    return {
        "patch_emb":       np.concatenate(patch_embs, axis=0),
        "sem_emb":         np.concatenate(sem_embs, axis=0),
        "sem_codes":       np.concatenate(sem_codes_list, axis=0),
        "var_labels":      np.concatenate(var_labels, axis=0),
        "pos_labels":      np.concatenate(pos_labels, axis=0),
        "var_labels_sem":  np.concatenate(var_labels_sem, axis=0),
    }


# ── plotting ─────────────────────────────────────────────────────────────────

def scatter(ax, xy, labels, title, cmap='tab20', label_prefix='', max_points=5000):
    if len(xy) > max_points:
        idx = np.random.choice(len(xy), max_points, replace=False)
        xy, labels = xy[idx], labels[idx]
    unique = np.unique(labels)
    colors = cm.get_cmap(cmap, len(unique))
    for i, u in enumerate(unique):
        mask = labels == u
        ax.scatter(xy[mask, 0], xy[mask, 1], s=4, alpha=0.4,
                   color=colors(i), label=f'{label_prefix}{u}' if len(unique) <= 21 else None)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    if len(unique) <= 21:
        ax.legend(markerscale=2, fontsize=5, loc='best', ncol=2)


def make_plots(data, out_dir, tsne_perplexity=30):
    os.makedirs(out_dir, exist_ok=True)

    patch_emb      = data["patch_emb"]
    sem_emb        = data["sem_emb"]
    sem_codes      = data["sem_codes"]
    var_labels     = data["var_labels"]
    pos_labels     = data["pos_labels"]
    var_labels_sem = data["var_labels_sem"]

    n_codes_used = len(np.unique(sem_codes))
    print(f"\nRepresentation stats:")
    print(f"  Patch embeddings shape : {patch_emb.shape}")
    print(f"  Semantic embeddings    : {sem_emb.shape}")
    print(f"  Unique VQ codes used   : {n_codes_used} / {config['codebook_size']}")
    print(f"  Code usage: {n_codes_used/config['codebook_size']*100:.1f}%")

    # ── PCA ─────────────────────────────────────────────────────────────────
    print("Running PCA on patch embeddings...")
    pca_patch = PCA(n_components=2).fit_transform(patch_emb[:10000])
    print("Running PCA on semantic embeddings...")
    pca_sem   = PCA(n_components=2).fit_transform(sem_emb[:10000])

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("PCA — Weather Representations", fontsize=12)

    scatter(axes[0, 0], pca_patch, var_labels[:len(pca_patch)],
            "Patch emb — colored by variable", cmap='tab20', label_prefix='var')
    scatter(axes[0, 1], pca_patch, pos_labels[:len(pca_patch)],
            "Patch emb — colored by patch position", cmap='viridis', label_prefix='pos')

    # variance per dim to assess collapse
    patch_var = patch_emb.var(axis=0)
    axes[0, 2].bar(range(len(patch_var)), np.sort(patch_var)[::-1])
    axes[0, 2].set_title("Patch emb: variance per dim (sorted)", fontsize=9)
    axes[0, 2].set_xlabel("Dimension (sorted)"); axes[0, 2].set_ylabel("Variance")

    scatter(axes[1, 0], pca_sem, var_labels_sem[:len(pca_sem)],
            "Semantic emb — colored by variable", cmap='tab20', label_prefix='var')
    scatter(axes[1, 1], pca_sem, sem_codes[:len(pca_sem)],
            f"Semantic emb — colored by VQ code ({n_codes_used} used)", cmap='tab20b', label_prefix='code')

    sem_var = sem_emb.var(axis=0)
    axes[1, 2].bar(range(len(sem_var)), np.sort(sem_var)[::-1])
    axes[1, 2].set_title("Semantic emb: variance per dim (sorted)", fontsize=9)
    axes[1, 2].set_xlabel("Dimension (sorted)"); axes[1, 2].set_ylabel("Variance")

    plt.tight_layout()
    pca_path = os.path.join(out_dir, "pca_weather.png")
    plt.savefig(pca_path, dpi=150)
    plt.close()
    print(f"PCA plot saved → {pca_path}")

    # ── t-SNE ───────────────────────────────────────────────────────────────
    max_tsne = 3000
    print(f"Running t-SNE on patch embeddings (n={min(max_tsne, len(patch_emb))})...")
    idx_p = np.random.choice(len(patch_emb), min(max_tsne, len(patch_emb)), replace=False)
    tsne_patch = TSNE(n_components=2, perplexity=tsne_perplexity,
                      n_iter=1000, random_state=42).fit_transform(patch_emb[idx_p])

    print(f"Running t-SNE on semantic embeddings (n={min(max_tsne, len(sem_emb))})...")
    idx_s = np.random.choice(len(sem_emb), min(max_tsne, len(sem_emb)), replace=False)
    tsne_sem = TSNE(n_components=2, perplexity=tsne_perplexity,
                    n_iter=1000, random_state=42).fit_transform(sem_emb[idx_s])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("t-SNE — Weather Representations", fontsize=12)

    scatter(axes[0, 0], tsne_patch, var_labels[idx_p],
            "Patch emb t-SNE — by variable", cmap='tab20', label_prefix='var')
    scatter(axes[0, 1], tsne_patch, pos_labels[idx_p],
            "Patch emb t-SNE — by patch position", cmap='viridis', label_prefix='pos')
    scatter(axes[1, 0], tsne_sem, var_labels_sem[idx_s],
            "Semantic emb t-SNE — by variable", cmap='tab20', label_prefix='var')
    scatter(axes[1, 1], tsne_sem, sem_codes[idx_s],
            f"Semantic emb t-SNE — by VQ code ({n_codes_used} used)", cmap='tab20b', label_prefix='code')

    plt.tight_layout()
    tsne_path = os.path.join(out_dir, "tsne_weather.png")
    plt.savefig(tsne_path, dpi=150)
    plt.close()
    print(f"t-SNE plot saved → {tsne_path}")

    # ── code histogram ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    counts = np.bincount(sem_codes, minlength=config["codebook_size"])
    ax.bar(range(len(counts)), counts, width=1.0)
    ax.set_title(f"VQ Code Usage Histogram — {n_codes_used}/{config['codebook_size']} codes used "
                 f"({n_codes_used/config['codebook_size']*100:.1f}%)", fontsize=11)
    ax.set_xlabel("Code index"); ax.set_ylabel("Count")
    hist_path = os.path.join(out_dir, "vq_code_histogram.png")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"Code histogram saved → {hist_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint .pt file. Auto-detected if omitted.")
    parser.add_argument("--out_dir", type=str, default="./output_model/DiscreteJEPA/repr_viz",
                        help="Directory to save plots.")
    parser.add_argument("--max_batches", type=int, default=30,
                        help="Number of test batches to collect embeddings from.")
    parser.add_argument("--tsne_perplexity", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── checkpoint ──────────────────────────────────────────────────────────
    ckpt_path = args.checkpoint or find_latest_checkpoint(config["path_save"].rstrip("/"))
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # ── build models ────────────────────────────────────────────────────────
    encoder = build_encoder()
    vq  = build_vq()

    encoder.load_state_dict(ckpt["target_encoder"])
    if "vector_quantizer_ema" in ckpt:
        vq.load_state_dict(ckpt["vector_quantizer_ema"])
    elif "vector_quantizer" in ckpt:
        vq.load_state_dict(ckpt["vector_quantizer"])

    encoder.to(device).eval()
    vq.to(device).eval()

    # ── weather data ────────────────────────────────────────────────────────
    print("Loading weather test data...")
    test_ds = ForcastingDataPullerDescrete(config, which='test')
    loader  = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)
    print(f"Test set: {len(test_ds)} samples, {len(loader)} batches")

    # ── collect embeddings ───────────────────────────────────────────────────
    print(f"Collecting embeddings from {args.max_batches} batches...")
    data = collect_embeddings(encoder, vq, loader, device, max_batches=args.max_batches)

    # ── plot ─────────────────────────────────────────────────────────────────
    make_plots(data, args.out_dir, tsne_perplexity=args.tsne_perplexity)
    print(f"\nDone. All plots saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
