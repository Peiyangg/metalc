import logging
import numpy as np
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from zadu import zadu

logging.getLogger('gensim').setLevel(logging.WARNING)


def evaluate_lda_pair(guided_tw, guided_dt, gensim_result, count_matrix, rel_matrix, feature_names, k_neighbors=10):
    """Evaluate a pair of Guided LDA and Standard LDA models.

    Parameters
    ----------
    guided_tw : (k, n_features) topic-word matrix from guidedlda
    guided_dt : (n_samples, k) doc-topic matrix from guidedlda
    gensim_result : dict with 'W' (n_features, k) and 'H' (k, n_samples) from run_lda
    count_matrix : (n_samples, n_features) integer counts
    rel_matrix : (n_samples, n_features) relative abundance
    feature_names : list of feature names
    k_neighbors : int, k for trustworthiness/continuity

    Returns
    -------
    dict with keys:
        'metrics_df': DataFrame with all scalar metrics
        'coherence_guided': list of per-topic C_v scores
        'coherence_standard': list of per-topic C_v scores
    """
    X_rel = np.asarray(rel_matrix)
    X_counts = np.asarray(count_matrix)

    std_tw = gensim_result['W'].T
    std_dt = gensim_result['H'].T

    guided_metrics = _compute_metrics(
        guided_tw, guided_dt, X_rel, X_counts, feature_names, k_neighbors, is_guided=True
    )
    std_metrics = _compute_metrics(
        std_tw, std_dt, X_rel, X_counts, feature_names, k_neighbors, is_guided=False,
        gensim_result=gensim_result
    )

    import pandas as pd
    rows = []
    for label, m in [("Guided LDA", guided_metrics), ("Standard LDA", std_metrics)]:
        rows.append({
            "Method": label,
            "Global Cosine": f"{m['global_cosine']:.4f}",
            "Per-Sample Cosine (mean)": f"{m['sample_cosine_mean']:.4f}",
            "Per-Sample Cosine (std)": f"{m['sample_cosine_std']:.4f}",
            "Perplexity": f"{m['perplexity']:.2f}",
            "KL Divergence": f"{m['kl_divergence']:.4f}",
            "Trustworthiness": f"{m['trustworthiness']:.4f}",
            "Continuity": f"{m['continuity']:.4f}",
            "Coherence (C_v)": f"{m['coherence_overall']:.4f}",
            "Coherence per topic": m['coherence_per_topic'],
        })

    return {
        'metrics_df': pd.DataFrame(rows),
        'coherence_guided': guided_metrics['coherence_per_topic'],
        'coherence_standard': std_metrics['coherence_per_topic'],
    }


def _compute_metrics(tw, dt, X_rel, X_counts, feature_names, k_neighbors, is_guided, gensim_result=None):
    recon = _reconstruct(tw, dt, is_guided, gensim_result)
    gc = _global_cosine(X_rel, recon)
    sm, ss = _per_sample_cosine(X_rel, recon)
    ppl = _perplexity(tw, dt, X_counts)
    kl, trust, cont = _dr_metrics(X_rel, dt, k_neighbors)
    c_overall, c_per_topic = _coherence(tw, X_counts, feature_names)

    return {
        'global_cosine': gc,
        'sample_cosine_mean': sm,
        'sample_cosine_std': ss,
        'perplexity': ppl,
        'kl_divergence': kl,
        'trustworthiness': trust,
        'continuity': cont,
        'coherence_overall': c_overall,
        'coherence_per_topic': c_per_topic,
    }


def _reconstruct(tw, dt, is_guided, gensim_result=None):
    if is_guided:
        recon = dt @ tw
    else:
        recon = (gensim_result['W'] @ gensim_result['H']).T
    row_sums = recon.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return recon / row_sums


def _global_cosine(X_true, X_recon):
    return np.sum(X_true * X_recon) / (np.linalg.norm(X_true) * np.linalg.norm(X_recon) + 1e-12)


def _per_sample_cosine(X_true, X_recon):
    sims = []
    for i in range(X_true.shape[0]):
        a, b = X_true[i], X_recon[i]
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        sims.append(np.dot(a, b) / (norm + 1e-12))
    return np.mean(sims), np.std(sims)


def _perplexity(tw, dt, X_counts):
    recon = np.clip(dt @ tw, 1e-12, None)
    total_ll = 0.0
    total_words = 0
    for i in range(X_counts.shape[0]):
        for j in range(X_counts.shape[1]):
            if X_counts[i, j] > 0:
                total_ll += X_counts[i, j] * np.log(recon[i, j])
                total_words += X_counts[i, j]
    return np.exp(-total_ll / total_words)


def _dr_metrics(X_rel, dt, k_neighbors):
    """KL divergence, trustworthiness, continuity via Zadu."""
    k = min(k_neighbors, dt.shape[0] - 1)
    spec = [
        {'id': 'tnc', 'params': {'k': k}},
        {'id': 'kl_div', 'params': {'sigma': 0.1}},
    ]
    z = zadu.ZADU(spec, X_rel, return_local=False)
    results = z.measure(dt)
    trust = float(results[0]['trustworthiness'])
    cont = float(results[0]['continuity'])
    kl = float(results[1]['kl_divergence'])
    return kl, trust, cont


def _coherence(tw, X_counts, feature_names, n_top=10):
    feat = [str(f) for f in feature_names]
    texts = []
    for i in range(X_counts.shape[0]):
        doc = []
        for j in range(X_counts.shape[1]):
            if X_counts[i, j] > 0:
                doc.extend([feat[j]] * int(X_counts[i, j]))
        texts.append(doc)
    dictionary = Dictionary(texts)

    k = tw.shape[0]
    topics_words = []
    for t in range(k):
        top_idx = tw[t].argsort()[-n_top:][::-1]
        topics_words.append([feat[j] for j in top_idx])

    cm = CoherenceModel(
        topics=topics_words, texts=texts, dictionary=dictionary, coherence='c_v'
    )
    per_topic = cm.get_coherence_per_topic()
    overall = cm.get_coherence()
    return overall, [round(v, 4) for v in per_topic]


def plot_eval_comparison(metrics_df, title, ax=None):
    """Small multiples bar chart comparing Guided vs Standard LDA metrics."""
    import matplotlib.pyplot as plt

    display_cols = [
        'Global Cosine', 'Per-Sample Cosine (mean)',
        'Trustworthiness', 'Continuity', 'Coherence (C_v)',
    ]
    inv_cols = ['Perplexity', 'KL Divergence']

    all_cols = display_cols + inv_cols
    n = len(all_cols)

    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3))
    methods = metrics_df['Method'].tolist()
    colors = ['#2196F3', '#FF9800']

    for i, col in enumerate(all_cols):
        vals = [float(metrics_df.iloc[j][col]) for j in range(len(methods))]
        axes[i].bar(methods, vals, color=colors[:len(methods)], width=0.5)
        axes[i].set_title(col, fontsize=9)
        axes[i].tick_params(axis='x', labelsize=7, rotation=30)
        axes[i].tick_params(axis='y', labelsize=7)
        if col in inv_cols:
            axes[i].set_title(f"{col} ↓", fontsize=9)
        else:
            axes[i].set_title(f"{col} ↑", fontsize=9)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig
