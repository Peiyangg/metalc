"""NMF-specific code: the sklearn GKL ensemble fitter, the fixed-W NMF solver,
and the Guided-vs-Standard track's plain-NMF primitives.

What lives here:
    run_nmf_ensemble              fit the NMF ensemble, pool + L1-normalise topics.
    infer_loadings_fixed_topics   fixed-W inference under GKL (sklearn), hold W.
    infer_topics_fixed_loadings   the symmetric solve — hold H, refit W.
    FixedWResult / infer_loadings  the older, NMF-only fixed-W + heatmap path
        (superseded by ``metalc.inference`` for new work, kept for the notebooks
        that still import it directly: ``Obesity_nmf_ensemble.py`` /
        ``PIG_nmf_ensemble.py``).
    run_guided_nmf, run_nmf, normalize, print_top_features
        Guided-vs-Standard track primitives (paired with ``evaluate.py``).

Everything method-agnostic (the ``Ensemble`` orchestration, clustering, caching,
plotting) lives in ``ensemble.py``; the LDA counterparts of everything here live
in ``lda.py``.
"""

import os

import numpy as np
import pandas as pd

from ssnmf import SSNMF

# --------------------------------------------------------------------------- #
# 1. Fit the ensemble
# --------------------------------------------------------------------------- #
#: NMF's iteration budget is a **ceiling, not a cost**: sklearn's `mu` solver
#: exits as soon as the loss stops improving by `tol`, so a high cap buys
#: headroom on tables that need it rather than time on tables that do not.
#: Measured on HRD (289x6866, GKL, tol=1e-4): 60-140 iterations at k=2,5,10 —
#: and identical `n_iter_` whether the cap was 500 or 20000, i.e. nothing was
#: truncating. It is set far above that so a bigger table (PIG's 15902 ASVs)
#: cannot silently stop short; `_report_nmf_convergence` says if one gets close.
MAX_ITER = 10000


def _report_nmf_convergence(diags, *, verbose=True):
    """Summarise an NMF ensemble's convergence, and warn on whatever hit the cap.

    The LDA twin's counterpart is ``lda._report_convergence``; the two report
    different quantities because the solvers expose different ones — sklearn
    has a real stopping rule and tells you when it did not reach it, gensim has
    neither and has to be read out of its logs. Neither drops a fit: an
    unsettled fit is one topic source among hundreds, and discarding it would
    bias the pool towards whichever k settles fastest.
    """
    import warnings

    if diags.empty:
        return diags
    n = len(diags)
    settled = diags["converged"].astype(bool)
    cap = int(diags["max_iter"].iloc[0])

    if verbose:
        ni = diags["n_iter"]
        print(f"Convergence (tol-based, max_iter={cap}): "
              f"{int(settled.sum())}/{n} fits settled", flush=True)
        print(f"  n_iter {ni.min()}-{ni.max()} (median {ni.median():.0f}), "
              f"{ni.max() / cap:.0%} of the ceiling at worst", flush=True)
        if not settled.all():
            by_k = diags.loc[~settled].groupby("k").size()
            print("  HIT THE CAP: "
                  + ", ".join(f"k={k} ({c} fits)" for k, c in by_k.items()),
                  flush=True)

    if not settled.all():
        warnings.warn(
            f"{int((~settled).sum())}/{n} NMF fits ran into max_iter={cap} "
            f"without meeting `tol` — those topics are wherever the solver "
            f"happened to be, not where it was heading. Raise `max_iter`. "
            f"Pooled topics from an unconverged fit add spread that the "
            f"clustering will read as structure.",
            RuntimeWarning, stacklevel=3,
        )
    return diags


def run_nmf_ensemble(
    V,
    feature_names,
    k_range=range(2, 11),
    n_runs=50,
    beta_loss="kullback-leibler",
    solver="mu",
    max_iter=MAX_ITER,
    tol=1e-4,
    seed_base=0,
    l1_normalize=True,
    progress=True,
):
    """Fit an NMF ensemble under generalized-KL loss and pool every topic.

    For each ``k`` in ``k_range`` and each of ``n_runs`` random initialisations,
    ``sklearn.decomposition.NMF`` factorises ``V (n_samples x n_features) ~= W H``
    with ``W (n_samples x k)`` sample loadings and ``H = components_ (k x n_features)``
    the topics. Every topic (every row of every run's ``components_``) is stacked
    into one long ``topic_feature_df``; ``topic_meta_df`` records which
    ``(k, run, topic)`` each row came from plus that topic's ``weight`` (its total
    loading mass across samples), so the downstream k-distribution / stability
    helpers work exactly as they do for the LDA ensemble.

    Parameters
    ----------
    V : array-like (n_samples, n_features)
        Non-negative counts / abundances (samples are rows, features are columns).
    feature_names : sequence of str, length n_features
        Column labels for ``topic_feature_df`` (e.g. genus names).
    k_range : iterable of int
        Numbers of components to sweep (default 2..10).
    n_runs : int
        Random initialisations per k.
    beta_loss, solver : str
        Passed to ``NMF``. ``'kullback-leibler'`` requires ``solver='mu'``.
    max_iter, tol : int, float
        A **ceiling and a tolerance**, not a cost: sklearn stops as soon as the
        loss settles to within ``tol``, so over-provisioning ``max_iter`` buys
        headroom rather than time. Measured need on HRD (289x6866) at k=2,5,10:
        60-140 iterations. Every fit records whether it stopped on ``tol`` or
        ran into the ceiling — see :func:`_report_nmf_convergence`.
    seed_base : int
        Base offset for the per-fit ``random_state = seed_base + k*1000 + run``,
        so the whole ensemble is reproducible.
    l1_normalize : bool
        L1-normalise each pooled topic to a probability vector (so JSD applies,
        matching the LDA pipeline). Keep True unless you cluster with cosine.

    Returns
    -------
    topic_feature_df : pd.DataFrame, shape (n_topics_total, n_features)
        One row per pooled topic; columns = ``feature_names``.
    topic_meta_df : pd.DataFrame
        Columns ``k, run, topic, weight`` aligned row-for-row with
        topic_feature_df, plus that fit's convergence record repeated on each of
        its topics: ``n_iter``, ``max_iter``, ``converged``,
        ``reconstruction_err``. It rides along in the meta frame so it survives
        ``ensemble.save_ensemble`` — a cached ensemble can be asked, later,
        whether it converged. (The LDA twin records the same thing under the
        names its own solver reports; see ``lda.py``.)
    """
    import time
    import warnings

    from sklearn.decomposition import NMF
    from sklearn.exceptions import ConvergenceWarning

    V = np.asarray(V, dtype=float)
    if np.any(V < 0):
        raise ValueError("V must be non-negative for NMF.")
    feature_names = list(feature_names)
    if V.shape[1] != len(feature_names):
        raise ValueError(
            f"feature_names length {len(feature_names)} != V.shape[1] {V.shape[1]}"
        )

    k_range = list(k_range)
    total = len(k_range) * n_runs
    rows = []
    meta = []
    diags = []
    done = 0
    t0 = time.time()

    for k in k_range:
        for run in range(n_runs):
            seed = seed_base + k * 1000 + run
            model = NMF(
                n_components=k,
                init="random",
                solver=solver,
                beta_loss=beta_loss,
                max_iter=max_iter,
                tol=tol,
                random_state=seed,
            )
            # sklearn signals "hit the cap" by raising ConvergenceWarning and
            # by leaving n_iter_ == max_iter. Both are checked: the warning is
            # the explicit signal, n_iter_ the one that survives a caller who
            # has filtered warnings off.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                W = model.fit_transform(V)      # (n_samples, k) loadings
                hit_cap = any(c.category is ConvergenceWarning for c in caught)
            H = model.components_               # (k, n_features) topics
            masses = W.sum(axis=0)              # topic usage across samples
            diags.append({
                "k": int(k), "run": int(run),
                "n_iter": int(model.n_iter_), "max_iter": int(max_iter),
                "converged": bool(not hit_cap and model.n_iter_ < max_iter),
                "reconstruction_err": float(model.reconstruction_err_),
            })

            for t in range(k):
                vec = H[t].astype(float)
                if l1_normalize:
                    s = vec.sum()
                    if s > 0:
                        vec = vec / s
                rows.append(vec)
                meta.append(
                    {"k": int(k), "run": int(run), "topic": int(t),
                     "weight": float(masses[t])}
                )

            done += 1
            if progress and (done % 25 == 0 or done == total or done == 1):
                elapsed = time.time() - t0
                rate = elapsed / done
                remaining = (total - done) * rate
                print(
                    f"[{done}/{total}] k={k} run={run} "
                    f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s left",
                    flush=True,
                )

    diag_df = pd.DataFrame(diags)
    _report_nmf_convergence(diag_df, verbose=progress)

    topic_feature_df = pd.DataFrame(rows, columns=feature_names).reset_index(drop=True)
    topic_meta_df = (pd.DataFrame(meta)
                     .merge(diag_df, on=["k", "run"], how="left")
                     .reset_index(drop=True))
    return topic_feature_df, topic_meta_df


# --------------------------------------------------------------------------- #
# 2. Fixed-factor inference under the SAME GKL loss
# --------------------------------------------------------------------------- #
def infer_loadings_fixed_topics(
    V,
    topics,
    beta_loss="kullback-leibler",
    solver="mu",
    max_iter=500,
    tol=1e-4,
):
    """Hold the topics (W) fixed; solve each sample's loadings H under GKL.

    This is the NMF form of fixed-W inference: once the meta-LC topics are
    chosen they are a fixed dictionary, and solving the per-sample loadings is
    a **convex** sub-problem (unique, reproducible). We solve it with
    ``sklearn.decomposition.non_negative_factorization(..., update_H=False)`` so
    the fixed factor is ``topics`` and only the loadings move — optimised with the
    *identical* ``beta_loss='kullback-leibler'`` used to fit the ensemble.

    There is no ``random_state``: with ``update_H=False`` sklearn ignores any
    supplied ``W`` and initialises it deterministically as a constant
    ``sqrt(X.mean() / n_topics)``, so this path carries no randomness at all.

    In sklearn's ``X ~= W_sk @ H_sk`` convention we pass ``X = V`` and
    ``H_sk = topics``; the returned ``W_sk`` is the loadings we want.

    Parameters
    ----------
    V : array-like (n_samples, n_features)
        The same count/abundance matrix the ensemble was fit on (or new samples
        with the same features, in the same column order).
    topics : array-like (n_topics, n_features)
        Fixed topics — e.g. the medoid or average meta-LC profiles from
        ``ensemble.build_profiles`` (``.values``). Rows are components, cols are
        features.

    Returns
    -------
    loadings : np.ndarray (n_samples, n_topics)
        Non-negative sample x meta-LC loadings.
    """
    from sklearn.decomposition import non_negative_factorization

    V = np.asarray(V, dtype=float)
    topics = np.asarray(topics, dtype=float)
    n_topics = topics.shape[0]
    if topics.shape[1] != V.shape[1]:
        raise ValueError(
            f"topics has {topics.shape[1]} features, V has {V.shape[1]}; "
            "they must share the same feature axis / column order."
        )

    loadings, _H, _n_iter = non_negative_factorization(
        V,
        W=None,               # solved for us; init='custom' only fixes H
        H=topics,
        n_components=n_topics,
        init="custom",
        update_H=False,
        solver=solver,
        beta_loss=beta_loss,
        max_iter=max_iter,
        tol=tol,
    )
    return loadings


def infer_topics_fixed_loadings(
    V,
    loadings,
    beta_loss="kullback-leibler",
    solver="mu",
    max_iter=500,
    tol=1e-4,
):
    """Symmetric solve: hold the loadings (H) fixed, refit the topics under GKL.

    The mirror of :func:`infer_loadings_fixed_topics` — useful if you fix a
    per-sample design and want the topics implied by it under the same loss.
    Solved by transposing (``V.T ~= topics.T @ loadings.T``) and fixing
    ``loadings.T`` as the sklearn ``H`` factor. Deterministic for the same
    reason as :func:`infer_loadings_fixed_topics` — no ``random_state``.

    Returns
    -------
    topics : np.ndarray (n_topics, n_features)
    """
    from sklearn.decomposition import non_negative_factorization

    V = np.asarray(V, dtype=float)
    loadings = np.asarray(loadings, dtype=float)
    n_topics = loadings.shape[1]
    if loadings.shape[0] != V.shape[0]:
        raise ValueError("loadings.shape[0] must equal V.shape[0] (n_samples).")

    topics_T, _H, _n_iter = non_negative_factorization(
        V.T,
        W=None,
        H=loadings.T,
        n_components=n_topics,
        init="custom",
        update_H=False,
        solver=solver,
        beta_loss=beta_loss,
        max_iter=max_iter,
        tol=tol,
    )
    return topics_T.T


# --------------------------------------------------------------------------- #
# 3. Old fixed-W + heatmap orchestration (NMF-only; superseded by
#    `metalc.inference`, kept for Obesity_nmf_ensemble.py / PIG_nmf_ensemble.py)
# --------------------------------------------------------------------------- #
class FixedWResult:
    """Return value of :func:`infer_loadings`."""

    def __init__(self, H, W, figure):
        self.H = H          # samples x meta-LC (row-normalised loadings)
        self.W = W          # meta-LC x feature (the fixed dictionary)
        self.figure = figure  # interactive lda4microbiome heatmap


def infer_loadings(ensemble, metadata, profile="medoid", lc_ids=None,
                   out_dir=None, order_by="heatmap_clustering",
                   universal_headers=None, continuous_headers=None):
    """Hold the meta-LC profiles fixed as W and infer each sample's H under GKL,
    then render the annotated heatmap.

    NMF-only: always solves with :func:`infer_loadings_fixed_topics`, regardless
    of ``ensemble.method``. Superseded by :func:`metalc.inference`, which
    dispatches on the method; kept because ``Obesity_nmf_ensemble.py`` /
    ``PIG_nmf_ensemble.py`` import it directly.

    Parameters
    ----------
    ensemble : ensemble.Ensemble
    metadata : pd.DataFrame
        Sample metadata (indexed by sample id, or with a column of sample ids).
        Categorical / continuous columns are auto-detected for the heatmap bars
        unless ``universal_headers`` / ``continuous_headers`` are given.
    profile : {"medoid", "average"}
        Which meta-LC profiles to freeze as W.
    lc_ids : list[int] or None
        Which meta-LCs to keep (e.g. ``[0, 3, 4]``). None/empty -> all of them.
    out_dir : str, optional
        Where to write the lda4microbiome layout + heatmap HTML. Defaults to a
        ``lda4microbiome_fixedW`` folder next to nothing in particular — pass one.

    Returns
    -------
    FixedWResult(H, W, figure)
    """
    import ensemble as _E
    import plotting as _PL
    from clustering import cluster_medoids

    labels = ensemble.labels
    medoid_indices = cluster_medoids(
        ensemble.topic_feature_df, labels, probabilities=ensemble.membership,
    )
    W_full = _E.build_profiles(
        ensemble.topic_feature_df, labels, medoid_indices,
        ensemble.n_clusters, kind=profile,
    )
    sel = list(range(W_full.shape[0])) if not lc_ids else list(lc_ids)
    W_df = W_full.iloc[sel]

    loadings = infer_loadings_fixed_topics(ensemble.df.values.astype(float), W_df.values)
    k = loadings.shape[1]
    rs = loadings.sum(axis=1, keepdims=True)
    Hn = loadings / np.where(rs == 0, 1.0, rs)
    # Samples the fixed W explains with zero loading get a uniform mixture rather
    # than an all-zero row (the heatmap clusters samples by cosine, undefined at 0).
    Hn[rs.ravel() == 0] = 1.0 / k
    H = pd.DataFrame(Hn, index=ensemble.sample_ids, columns=list(W_df.index))

    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), "lda4microbiome_fixedW")
    fig = _PL.render_fixedw_heatmap(
        H.values, W_df, ensemble.sample_ids, metadata, out_dir,
        group_map=ensemble.group_map, order_by=order_by,
        universal_headers=universal_headers, continuous_headers=continuous_headers,
    )
    return FixedWResult(H, W_df, fig)


# --------------------------------------------------------------------------- #
# 4. Guided-vs-Standard track primitives (paired with evaluate.py)
# --------------------------------------------------------------------------- #
def run_guided_nmf(X, feature_names, seed_features_list, k, lam=20, numiters=50, seed=1):
    """
    X : array (n_features, n_samples)
    feature_names : array-like of feature names, length n_features
    seed_features_list : list of dicts, e.g. [{"genusA": 1}, {"genusB": 1}]
    k : int, number of topics (>= len(seed_features_list) + 1)
    """
    feature_list = list(feature_names)
    topic_vectors = []
    for d in seed_features_list:
        v = np.zeros(len(feature_list))
        for name, weight in d.items():
            v[feature_list.index(name)] = weight
        topic_vectors.append(v)
    Y = np.stack(topic_vectors).T

    np.random.seed(seed)
    model = SSNMF(X.T, k, Y=Y.T, lam=lam)
    model.snmfmult(numiters=numiters)

    W = model.S.T  # (n_features, k)
    H = model.A.T  # (k, n_samples)
    return {'W': W, 'H': H, 'B': model.B, 'model': model}


def run_nmf(X, k, numiters=50, seed=1, eps=1e-10):
    """
    X : array (n_features, n_samples)
    k : int, number of topics
    """
    np.random.seed(seed)
    m, n = X.shape
    W = np.random.rand(m, k)
    H = np.random.rand(k, n)

    for _ in range(numiters):
        W = W * (X @ H.T) / (W @ H @ H.T + eps)
        H = H * (W.T @ X) / (W.T @ W @ H + eps)

    return {'W': W, 'H': H}


def normalize(result, cluster=False):
    """Post-hoc normalization: W columns to unit L1, scale into H, then H columns sum to 1.
    If cluster=True, reorder samples by dominant topic, then by weight within each topic."""
    W = result['W'].copy()
    H = result['H'].copy()
    col_norms = W.sum(axis=0, keepdims=True)
    col_norms[col_norms == 0] = 1
    W = W / col_norms
    H = H * col_norms.T
    h_col_sums = H.sum(axis=0, keepdims=True)
    h_col_sums[h_col_sums == 0] = 1
    H = H / h_col_sums
    if cluster:
        dominant = np.argmax(H, axis=0)
        weight = np.max(H, axis=0)
        order = np.lexsort((-weight, dominant))
        H = H[:, order]
        return W, H, order
    return W, H


def print_top_features(result, feature_names, n_top=10, display_names=None):
    """Print top features per topic using normalized W.
    If display_names is provided, aggregate weights by display name (genus level)."""
    W, H = normalize(result)
    raw_W = result['W']
    raw_col_sums = raw_W.sum(axis=0)
    topic_share = H.sum(axis=1)
    topic_share = topic_share / (topic_share.sum() + 1e-12)
    feature_names = np.asarray(feature_names)
    if display_names is not None:
        show = np.asarray(display_names)
        for i in range(W.shape[1]):
            genus_weights = {}
            for j in range(W.shape[0]):
                g = show[j]
                genus_weights[g] = genus_weights.get(g, 0) + W[j, i]
            ranked = sorted(genus_weights.items(), key=lambda x: x[1], reverse=True)[:n_top]
            parts = [f"{name} ({w:.3f})" for name, w in ranked]
            print(f"  Topic {i+1} [share={topic_share[i]:.1%}, raw_sum={raw_col_sums[i]:.2f}]: {', '.join(parts)}")
    else:
        for i in range(W.shape[1]):
            top_idx = W[:, i].argsort()[-n_top:][::-1]
            weights = W[top_idx, i]
            parts = [f"{feature_names[j]} ({weights[idx]:.3f})" for idx, j in enumerate(top_idx)]
            print(f"  Topic {i+1} [share={topic_share[i]:.1%}, raw_sum={raw_col_sums[i]:.2f}]: {', '.join(parts)}")
