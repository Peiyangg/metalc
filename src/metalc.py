"""metalc — Consensus Latent Components. The public API.

    import metalc as mlc

    ensemble = mlc.run_ensemble(df, [2, 10], method="lda",
                                group_map=asv_to_genus,
                                cache_dir=".../lda_ensemble_results")

    ensemble.layers()                     # every PLSCAN resolution
    e = ensemble[2]                       # fix the layer ONCE

    mlc.plot_meta(e, view="profile")      # top taxa per meta-LC
    mlc.plot_meta(e, view="umap")         # UMAP overview

    result = mlc.inference(e, metadata, meta_lc_list=[0, 3, 4])
    result.H                              # samples x meta-LC
    result.W                              # meta-LC x feature (the frozen dictionary)

    from lda4microbiome import a_heatmap
    a_heatmap(result.H, metadata, sort=...)

The pipeline: fit ``50`` models at every ``k``, pool every topic, map them to
``sqrt(p)`` (where euclidean *is* Hellinger), cluster with PLSCAN into
**meta-LCs**, then freeze chosen meta-LC profiles as **W** and solve only **H**.
Nothing is refit at the last step.

Two knobs must agree between the picture and the inference, and both are
therefore pinned to the object rather than passed ad hoc:

* **the layer** — meta-LC ids are numbered *per layer*, so ``LC-0`` at layer 2
  and ``LC-0`` at the best layer are different clusters. Select once with
  ``ensemble[i]`` and pass that object everywhere.
* **the method** — recorded by :func:`run_ensemble`, and :func:`inference`
  dispatches on it. An LDA ensemble folded in under NMF-GKL would silently drop
  the Dirichlet prior.

Everything here is a thin façade over ``ensemble`` (method-agnostic
orchestration) / ``lda`` / ``nmf`` (the two fitters' own math) /
``clustering``, which keep their existing names and signatures.
"""

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

import ensemble as _P
import lda as L
import nmf as N
from ensemble import Ensemble, ProfilePlot
from clustering import (
    cluster_medoids,
    embed_topics,
    cluster_topics,
    cluster_topics_stable,
    hellinger_embed,
    layer_table,
    project_topics,
)

__all__ = [
    "run_ensemble", "plot_meta", "inference", "InferenceResult",
    "Ensemble", "ProfilePlot",
    "cluster_topics", "cluster_topics_stable", "cluster_medoids", "embed_topics",
    "hellinger_embed", "project_topics", "layer_table",
]

_VIEWS = {"profile", "umap"}
_MANIFEST = "metalc_manifest.json"
_CLUSTERING = "clustering.npz"


class _CachedClusterer:
    """A restored PLSCAN result — only the three attributes anything reads.

    Not a PLSCAN: it cannot re-fit or expose a condensed tree. It exists so a
    cached clustering can stand in for the real thing everywhere ``Ensemble``
    looks (``cluster_layers_``, ``layer_persistence_scores_``,
    ``membership_strengths_``).
    """

    def __init__(self, layers, persistence, membership):
        self.cluster_layers_ = [np.asarray(l) for l in layers]
        self.layer_persistence_scores_ = np.asarray(persistence)
        self.membership_strengths_ = np.asarray(membership)


def _check_fit_budget(manifest, method, cache_dir, fit_params=None):
    """Refuse pooled topics whose **stopping rule** differs from this build's.

    Topics are the expensive artefact and are reused on sight, so a change to
    when a fit is allowed to stop would otherwise be invisible: the ensemble
    would keep answering with topics nobody's settings produce any more.

    A raised *ceiling* is not such a change. ``max_passes`` / ``max_iter`` bound
    a loop that exits on its own, so raising one can only let a fit run longer —
    and if nothing hit the old ceiling the topics are identical. That case
    warns instead of refusing, because the alternative is forcing a refit of
    hundreds of fits to reproduce the same numbers. (Measured: no HRD NMF fit
    gets past 170 iterations at k=2..20, against the old cap of 500.)
    """
    want = _P.fit_budget(method, fit_params)
    have = manifest.get("fit") or _P.legacy_fit_budget(method)
    # Compared key by key, a missing key reading as None: a budget key added
    # later (``move_tol``) defaults to None, and a manifest written before it
    # existed was fitted under exactly that default.
    if all(have.get(k) == want.get(k) for k in set(want) | set(have)):
        return

    rule = sorted(k for k in set(want) | set(have)
                  if k not in _P.CEILING_KEYS and have.get(k) != want.get(k))
    if rule:
        raise ValueError(
            f"{cache_dir!r} holds topics fitted under a different stopping rule"
            + ("" if manifest.get("fit") else
               " (no recorded budget — it predates convergence tracking)")
            + ": " + ", ".join(f"{k}={have.get(k)!r} on disk vs {want[k]!r} now"
                               for k in rule)
            + ". Refit the ensemble, or point `cache_dir` at a fresh directory. "
              "To keep the cached topics deliberately, set "
              f'"fit": {json.dumps(want)} in {_MANIFEST} there.'
        )

    lowered = [k for k in _P.CEILING_KEYS
               if k in want and (want[k] or 0) < (have.get(k) or 0)]
    if lowered:
        raise ValueError(
            f"{cache_dir!r} holds topics fitted with a *higher* ceiling than "
            f"this build allows ({', '.join(f'{k}: {have[k]} -> {want[k]}' for k in lowered)}). "
            "A fit that used the old headroom cannot be reproduced now."
        )
    import warnings
    warnings.warn(
        f"{cache_dir!r} was fitted with a lower ceiling than this build uses "
        + ", ".join(f"{k}: {have.get(k)} -> {want[k]}"
                    for k in sorted(_P.CEILING_KEYS & set(want))
                    if have.get(k) != want[k])
        + ". The cached topics are reused: raising a ceiling only matters for a "
          "fit that hit the old one. Ensembles fitted from now on record per-fit "
          "convergence in `topic_meta_df`, so this is checkable rather than "
          "assumed.",
        RuntimeWarning, stacklevel=3,
    )


def _clustering_key(embed, embed_dim, umap_random_state, n_repeats, n_topics):
    """What a cached clustering is only valid for.

    Every one of these changes the partition, so a mismatch must invalidate
    rather than silently hand back labels formed under other settings.
    """
    return {"embed": embed, "embed_dim": int(embed_dim),
            "umap_random_state": int(umap_random_state),
            "n_repeats": int(n_repeats), "n_topics": int(n_topics)}


def _load_clustering(cache_dir, key):
    """Cached (clusterer, votes, embedding) if it matches ``key``, else None."""
    path = os.path.join(cache_dir, _CLUSTERING)
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    if json.loads(str(z["key"])) != key:
        return None
    emb = z["embedding"] if "embedding" in z.files else None
    return (_CachedClusterer(list(z["layers"]), z["persistence"], z["membership"]),
            json.loads(str(z["votes"])), emb)


def _save_clustering(cache_dir, clusterer, votes, key, embedding):
    """Persist the modal partition so later sessions get the *same* meta-LC ids.

    The embedding is stored only when it is small and not reproducible for free
    — i.e. the UMAP path. ``sqrt(p)`` is deterministic and ~150MB, so it is
    recomputed instead (about 2s) rather than cached.
    """
    payload = {
        "layers": np.asarray([np.asarray(l) for l in clusterer.cluster_layers_]),
        "persistence": np.asarray(clusterer.layer_persistence_scores_),
        "membership": np.asarray(clusterer.membership_strengths_),
        "votes": json.dumps(votes, default=float),
        "key": json.dumps(key),
    }
    if embedding is not None and embedding.shape[1] <= 64:
        payload["embedding"] = np.asarray(embedding, dtype=np.float32)
    np.savez_compressed(os.path.join(cache_dir, _CLUSTERING), **payload)


def run_ensemble(df=None, k_range=None, method="nmf", taxonomy=None,
                 cache_dir=None, n_runs=50, n_repeats=10, workers=1,
                 fit_params=None,
                 embed="hellinger", embed_dim=10, umap_random_state=42,
                 recluster=False):
    """Fit an ensemble, cluster the pooled topics into meta-LCs, and cache everything.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Abundance matrix, **rows = samples, columns = features** (ASVs). Required
        unless ``cache_dir`` already holds a completed run. Both fitters take the
        same matrix; counts are the standard input (LDA's multinomial model
        assumes them and GKL is the matching NMF loss), but nothing is enforced.
    k_range : [lo, hi] or iterable, optional
        ``[2, 10]`` means 2..10 inclusive. Required unless loading from cache.
    method : {"nmf", "lda"}
        Which fitter produces the pooled topics. Recorded on the result and used
        by :func:`inference` to pick the matching fixed-W solver.
    taxonomy : pd.DataFrame, optional
        Annotation table **indexed by feature id, matching ``df.columns``** — for
        ASV data this is the taxonomy file (``Feature ID`` as index, ranks as
        columns). Nothing about the fitting changes: it stays in feature space.
        It is used only by :func:`plot_meta`, which needs you to name the column
        to annotate at. Any feature-indexed table works, so non-ASV data (e.g.
        CPR) can supply whatever grouping it has.
    cache_dir : str, optional
        Where to cache the run. Everything needed to reconstruct the ensemble is
        written here — pooled topics, the input matrix, the taxonomy, and a
        manifest of ``method`` / ``n_runs`` / ``k_range`` — so a later
        ``run_ensemble(cache_dir=...)`` needs no other argument. Fitting is
        skipped on a cache hit; PLSCAN is always re-run (it is cheap, and the
        vote must be fresh).
    n_runs : int
        Fits per k. Lower it only for smoke tests.
    n_repeats : int
        PLSCAN fits to vote over. PLSCAN is not reproducible, so the clustering
        is the mode of several fits rather than a single draw.
    workers : int
        Parallel fitting processes (``method="lda"`` only).
    fit_params : dict, optional
        Overrides forwarded to the fitter, and recorded in the manifest so the
        cache still knows what produced it — e.g. ``{"rel_tol": 1e-4}`` for a
        stricter LDA stop, or ``{"passes": 100}`` to reproduce an ensemble
        fitted before the early stop existed. The LDA stop works for any table
        size: one larger than ``chunksize`` (SO-CPR's 55,886 segments) is fitted
        online, several M-steps per pass, and the stop judges whole passes.
    embed : {"hellinger", "umap"}
        Which space PLSCAN clusters in. ``"hellinger"`` (default) is ``sqrt(p)``
        itself, deterministic and parameter-free. ``"umap"`` clusters an
        ``embed_dim``-D UMAP of that same array under euclidean — the only
        metric a UMAP output has — which still gives PLSCAN its exact KDTree MST
        at 10 dimensions. It is deterministic given ``umap_random_state``, but it
        makes the meta-LC ids depend on that seed and on the UMAP version, and it
        puts a fitted, lossy step between the topics and the clusters. See
        :func:`clustering.embed_topics`.
    embed_dim, umap_random_state : int
        For ``embed="umap"``. ``umap_random_state`` is also the 2-D overview's
        seed, so the clustering UMAP and the picture share a kNN graph (not a
        layout).
    recluster : bool
        Re-run the PLSCAN vote even when ``cache_dir`` holds one. The clustering
        is cached alongside the topics, because PLSCAN is not reproducible: a
        fresh vote can renumber — or re-count — the layers, which silently
        re-points any meta-LC id already recorded. The cache is keyed on
        ``embed`` / ``embed_dim`` / ``umap_random_state`` / ``n_repeats`` and the
        topic count, so changing any of those invalidates it on its own.

    Returns
    -------
    Ensemble — defaults to the highest-persistence PLSCAN layer. Fix a layer with
    ``ensemble[i]`` before doing anything downstream.
    """
    # Two independent caches, and they must be checked separately. The pooled
    # topics are the expensive artefact; the manifest only records how they were
    # made. A directory written by `ensemble.save_ensemble` directly (as the
    # HRD results were) has topics but no manifest — treating that as a miss
    # would silently refit the whole ensemble.
    manifest_path = os.path.join(cache_dir, _MANIFEST) if cache_dir else None
    has_manifest = bool(manifest_path and os.path.exists(manifest_path))
    has_topics = bool(cache_dir and any(
        os.path.exists(os.path.join(cache_dir, f))
        for f in ("all_topics_feature.npy", "all_topics_feature.csv")
    ))

    if has_manifest:
        with open(manifest_path) as fh:
            man = json.load(fh)
        method, n_runs = man["method"], man["n_runs"]
        k_range = k_range or man["k_range"]
        _check_fit_budget(man, method, cache_dir, fit_params)
        if df is None:
            df = _load_table(cache_dir)
        if taxonomy is None:
            tax_path = os.path.join(cache_dir, "taxonomy.csv")
            if os.path.exists(tax_path):
                taxonomy = pd.read_csv(tax_path, index_col=0)
    elif df is None or k_range is None:
        raise ValueError(
            f"df and k_range are required: {cache_dir!r} has no {_MANIFEST}."
            + (" Its pooled topics will still be reused — this call only needs "
               "the inputs once, and writes the manifest so later calls can pass "
               "cache_dir alone." if has_topics else "")
        )

    if taxonomy is not None:
        _check_taxonomy(taxonomy, df)

    # Third cache, independent of the other two: the clustering. PLSCAN is not
    # reproducible, so re-voting each session can renumber the meta-LCs an
    # analyst has already selected by id -- and on HRD it costs ~4 minutes at
    # the default n_repeats. Reusing it keeps ids stable *and* fast.
    clustering = None
    if cache_dir and not recluster and has_topics:
        try:
            tf, _tm, _ = _P.load_ensemble(cache_dir)
            clustering = _load_clustering(
                cache_dir,
                _clustering_key(embed, embed_dim, umap_random_state,
                                n_repeats, len(tf)),
            )
        except FileNotFoundError:
            pass

    # cache_dir goes straight through: the fitter reuses the pooled topics when
    # they are there and saves them when they are not.
    ensemble = _P.run_ensemble(
        df, k_range, method=method, cache_dir=cache_dir,
        n_repeats=n_repeats, workers=workers, n_runs=n_runs,
        embed=embed, embed_dim=embed_dim, umap_random_state=umap_random_state,
        clustering=clustering, fit_params=fit_params,
    )
    ensemble.taxonomy = taxonomy

    if cache_dir:
        # Top up whatever the directory is missing, so the next call needs no
        # arguments at all. Never overwrites the topics themselves.
        if not os.path.exists(os.path.join(cache_dir, "input_table.npy")):
            np.save(os.path.join(cache_dir, "input_table.npy"),
                    df.values.astype(np.float32))
            pd.DataFrame({"sample_id": list(df.index)}).to_csv(
                os.path.join(cache_dir, "sample_ids.csv"), index=False)
        if taxonomy is not None and not os.path.exists(
                os.path.join(cache_dir, "taxonomy.csv")):
            taxonomy.to_csv(os.path.join(cache_dir, "taxonomy.csv"))
        if not has_manifest:
            with open(manifest_path, "w") as fh:
                json.dump({"method": method, "n_runs": n_runs,
                           "k_range": list(_P._as_krange(k_range)),
                           "fit": _P.fit_budget(method, fit_params)}, fh, indent=2)
        if clustering is None:
            _save_clustering(
                cache_dir, ensemble.clusterer, ensemble.votes,
                _clustering_key(embed, embed_dim, umap_random_state, n_repeats,
                                len(ensemble.topic_feature_df)),
                ensemble.embedding,
            )
    return ensemble


def _load_table(cache_dir):
    """The cached input matrix, rebuilt with its sample and feature labels."""
    X = np.load(os.path.join(cache_dir, "input_table.npy"))
    rows = pd.read_csv(os.path.join(cache_dir, "sample_ids.csv"))["sample_id"]
    cols = pd.read_csv(os.path.join(cache_dir, "feature_ids.csv"))["feature_id"]
    return pd.DataFrame(X, index=list(rows), columns=list(cols))


def _check_taxonomy(taxonomy, df):
    """The taxonomy is indexed by **feature**, not by sample — catch the mix-up here.

    Getting this backwards silently produces an all-missing annotation, so it is
    worth an explicit error naming the likely cause.
    """
    if not isinstance(taxonomy, pd.DataFrame):
        raise TypeError(
            f"taxonomy must be a DataFrame indexed by feature id, got {type(taxonomy).__name__}."
        )
    if df is None:
        return
    hit = len(set(taxonomy.index) & set(df.columns))
    if hit == 0:
        looks_like_samples = bool(set(taxonomy.index) & set(df.index))
        raise ValueError(
            "taxonomy.index matches none of df.columns. The taxonomy must be "
            "indexed by **feature id** (ASV), which are df's columns"
            + (" — the index given looks like sample ids, which are df's rows."
               if looks_like_samples else ".")
        )


def _profiles(ensemble, group_map=None, cache_key=""):
    """The ensemble's :class:`ProfilePlot`, built once and cached on the object.

    ``ProfilePlot.__init__`` eagerly computes the medoids and *both* profile
    kinds, so rebuilding it per view would repeat that work. The cache is keyed
    on the selected layer *and* the annotation rank: ``ensemble[i]`` returns a
    shallow copy, so a different layer gets its own cache rather than inheriting
    a stale one, and switching rank must not reuse the previous aggregation.
    """
    key = f"_profileplot_L{ensemble.selected_layer}_{cache_key}"
    pp = getattr(ensemble, key, None)
    if pp is None:
        pp = ProfilePlot(ensemble, group_map=group_map)
        setattr(ensemble, key, pp)
    return pp


def plot_meta(ensemble, view="profile", rank=None, profile="medoid",
              coherence=False, **kwargs):
    """Draw one of the two meta-LC views for the ensemble's **selected layer**.

    Parameters
    ----------
    ensemble : Ensemble
        Already fixed to the layer you want (``e = ensemble[2]``). Both views
        read the same layer, so they cannot disagree.
    view : {"profile", "umap"}
        ``"profile"`` — small-multiples of each meta-LC's top features.
        ``"umap"`` — UMAP overview coloured by meta-LC. The projection runs in
        the *same* Hellinger geometry the clustering used, so a point can never
        appear inside the wrong cluster's colour.
    rank : str
        Which ``taxonomy`` column to annotate at, e.g. ``"Genus"``. **Required
        when the ensemble carries a taxonomy, with no default** — the analysis
        runs at feature (ASV) level, and the level you read the profiles at is a
        deliberate choice, not something to inherit silently. Omit it (or pass
        ``None``) to stay at feature level.
    profile : {"medoid", "average"}
        Which profile the view describes. ``"medoid"`` is a real topic row;
        ``"average"`` is the cluster mean. **Use the same value you pass to**
        :func:`inference` — otherwise the figure describes different vectors
        from the ones frozen as W.
    coherence : bool
        Print each meta-LC's ``u_mass`` coherence beside its row label
        (``"profile"`` view only). Scores are negative; **less negative is more
        coherent**, and they compare meta-LCs only within this one corpus.
        Always computed in feature space against the fitted matrix, whatever
        ``rank`` the profile is displayed at — see
        :func:`ensemble.topic_coherence_umass`.
    **kwargs
        Forwarded to the underlying plot: ``order_by``, ``feature_cutoff``,
        ``title``, ``save_path`` for ``"profile"``; ``umap_random_state``,
        ``feature_cutoff``, ``title``, ``sub_title``, ``save_path`` for ``"umap"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if view not in _VIEWS:
        raise ValueError(f'view must be one of {sorted(_VIEWS)}, got {view!r}')
    if profile not in ("medoid", "average"):
        raise ValueError(f'profile must be "medoid" or "average", got {profile!r}')

    taxonomy = getattr(ensemble, "taxonomy", None)
    group_map, cache_key = ensemble.group_map, "raw"
    if taxonomy is not None:
        if rank is None:
            raise ValueError(
                "rank is required when the ensemble has a taxonomy: name the "
                f"column to annotate at, e.g. rank='Genus'. Available: "
                f"{list(taxonomy.columns)}. Pass the ensemble without a taxonomy "
                "to stay at feature level."
            )
        if rank not in taxonomy.columns:
            raise KeyError(
                f"rank {rank!r} is not a taxonomy column. Available: "
                f"{list(taxonomy.columns)}"
            )
        group_map = taxonomy[rank].to_dict()
        cache_key = str(rank)
    elif rank is not None:
        raise ValueError(
            f"rank={rank!r} given but the ensemble has no taxonomy. Pass "
            "taxonomy= to run_ensemble first."
        )

    pp = _profiles(ensemble, group_map=group_map, cache_key=cache_key)
    if view != "profile":
        if coherence:
            raise ValueError("coherence is only available for view='profile'.")
        return pp.projection(profile=profile, **kwargs)

    if coherence:
        # Scored on the *feature-level* profile, not the displayed (aggregated)
        # one: the corpus is in feature space, and the rank is only a display
        # choice. Keyed by the row labels plot_cluster_profile draws.
        scores = _P.topic_coherence_umass(
            pp._feature_profile[profile], ensemble.df.values,
            ensemble.feature_names,
        )
        kwargs.setdefault(
            "row_annotations",
            {k: f"u_mass {v:+.2f}" for k, v in scores.items()})
    label = f" ({rank})" if rank else ""
    kwargs.setdefault(
        "title",
        f"{ensemble.method.upper()} meta-LC profiles{label} — {profile.upper()}")
    return pp.top_taxa(profile=profile, **kwargs)


@dataclass
class InferenceResult:
    """Return value of :func:`inference`.

    Deliberately carries no figure: solving H is a cheap, deterministic,
    one-shot convex problem, while the heatmap gets redrawn with different
    metadata columns and sort orders. Hand ``H`` straight to
    ``lda4microbiome.a_heatmap(result.H, metadata, ...)``.

    ``H`` rows always sum to 1, but that is *free* for only one of the two
    methods. Because every W row is a distribution,
    ``sum_j H_ij = sum_f (WH)_if ~= sum_f V_if`` — the sample's library size.
    LDA's ``transform`` already returns the normalised mixture (rows sum to 1
    natively, so normalising is a no-op), whereas the NMF loadings come back
    scaled by sequencing depth and the normalisation is what turns them into
    proportions. Verified on HRD: NMF row sums equalled the library sizes
    exactly, LDA's were 1.0.
    """
    H: pd.DataFrame          # samples x meta-LC, rows sum to 1
    W: pd.DataFrame          # meta-LC x feature, the frozen dictionary
    method: str              # which fitter's fixed-W solver produced H


def inference(ensemble, metadata=None, meta_lc_list=None, profile="medoid",
              alpha=None):
    """Freeze the selected meta-LC profiles as **W** and solve each sample's **H**.

    Nothing is refit. With W held fixed the problem is convex, so H is unique
    and reproducible. The solver follows the ensemble's own ``method``:

    * ``"nmf"`` — ``non_negative_factorization(..., update_H=False)`` under the
      same generalized-KL loss the ensemble was fitted with.
    * ``"lda"`` — sklearn's variational E-step with the topics injected as
      ``components_``, i.e. ``LatentDirichletAllocation.transform``. The
      Dirichlet ``alpha`` prior is what regularises H against a redundant
      dictionary, which is why the two paths are not interchangeable.

    Parameters
    ----------
    ensemble : Ensemble
        Fixed to the layer whose meta-LC ids you are about to name
        (``e = ensemble[2]``). Ids are per-layer, so passing ids read off a
        figure drawn at another layer would freeze the wrong topics.
    metadata : pd.DataFrame, optional
        Sample metadata. Used only to align the output: ``H`` is returned in
        ``metadata``'s row order, so it can go straight into
        ``a_heatmap(result.H, metadata)``. A mismatch in sample ids raises
        rather than silently mis-pairing rows.
    meta_lc_list : list[int], optional
        Which meta-LCs to keep, e.g. ``[0, 3, 4]``. ``None`` or ``[]`` keeps all.
    profile : {"medoid", "average"}
        Which profiles to freeze. Match what you passed to :func:`plot_meta`.
    alpha : float, optional
        ``doc_topic_prior`` for the LDA path; ignored for NMF. ``None`` uses
        sklearn's default ``1/n_topics``.

    Returns
    -------
    InferenceResult(H, W, method)
    """
    if profile not in ("medoid", "average"):
        raise ValueError(f'profile must be "medoid" or "average", got {profile!r}')

    labels = ensemble.labels
    medoid_indices = cluster_medoids(
        ensemble.topic_feature_df, labels, probabilities=ensemble.membership,
    )
    W_full = _P.build_profiles(
        ensemble.topic_feature_df, labels, medoid_indices,
        ensemble.n_clusters, kind=profile,
    )
    sel = list(range(W_full.shape[0])) if not meta_lc_list else list(meta_lc_list)
    bad = [i for i in sel if not 0 <= i < W_full.shape[0]]
    if bad:
        raise IndexError(
            f"meta-LC {bad} out of range: layer {ensemble.selected_layer} has "
            f"{W_full.shape[0]} meta-LCs (0..{W_full.shape[0] - 1}). Ids are "
            "numbered per layer — check you read them off this layer's figure."
        )
    if len(sel) < 2:
        raise ValueError(
            f"meta_lc_list selected {len(sel)} meta-LC; at least 2 are needed. "
            "H is a per-sample mixture normalised to sum to 1, so a single "
            "component makes every row exactly 1.0 and carries no information."
        )
    W_df = W_full.iloc[sel]

    V = ensemble.df.values.astype(float)
    if ensemble.method == "lda":
        loadings = L.infer_loadings_fixed_topics_lda(V, W_df.values, alpha=alpha)
    else:
        loadings = N.infer_loadings_fixed_topics(V, W_df.values)

    k = loadings.shape[1]
    rs = loadings.sum(axis=1, keepdims=True)
    Hn = loadings / np.where(rs == 0, 1.0, rs)
    # A sample the fixed W explains with zero loading gets a uniform mixture
    # rather than an all-zero row — downstream heatmaps cluster samples by
    # cosine, which is undefined at the origin.
    Hn[rs.ravel() == 0] = 1.0 / k
    H = pd.DataFrame(Hn, index=ensemble.sample_ids, columns=list(W_df.index))

    if metadata is not None:
        idx = metadata.index
        missing = [s for s in idx if s not in H.index]
        if missing:
            raise KeyError(
                f"{len(missing)} metadata rows have no matching sample in the "
                f"ensemble (first few: {missing[:5]}). Align the ids before "
                "inference rather than letting the heatmap pair rows by position."
            )
        H = H.loc[idx]

    return InferenceResult(H=H, W=W_df, method=ensemble.method)
