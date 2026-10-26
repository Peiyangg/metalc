"""One clustering geometry, shared by the NMF and the LDA ensemble tracks.

Both tracks pool topics from many fits and cluster them into meta-LCs, but they
used to do it in *different* geometries, which made their meta-LCs incomparable:

    NMF  ->  PLSCAN on the topic x feature matrix, metric='jensen_shannon'
    LDA  ->  HDBSCAN on a 2-D UMAP embedding, metric='euclidean'

This module is the single answer both now call.

The geometry: Hellinger
-----------------------
Topics are probability vectors over features. Taking the elementwise square root
maps them onto the positive unit L2 sphere, where **plain euclidean distance is
the Hellinger distance** (up to a constant)::

    ||sqrt(p) - sqrt(q)||_2  =  sqrt(2) * H(p, q)

So ``PLSCAN(metric='euclidean')`` on ``sqrt(p)`` clusters under a proper metric
for distributions — closely related to the Jensen-Shannon distance the pipeline
was designed around — while letting PLSCAN take its **exact** KDTree/Boruvka MST
path. Non-euclidean metrics fall back to an approximate pynndescent kNN graph.

Why that matters, measured on the HRD NMF ensemble (2700 topics x 6866 ASVs):

    euclidean on sqrt(p)  [KDTree]        0.3 s     ARI 0.963 vs the old labels
    hellinger  [pynndescent]              3.5 s     ARI 0.964
    jensen_shannon  [pynndescent, old]   19.5 s     --

The first two agree at ARI 0.998, exactly as the identity above predicts. The
euclidean path is ~65x faster than the metric it replaces *and* it is stable:
three identical fits of the same data gave

    jensen_shannon        [61, 37, 17, 2]  [61, 37, 17, 2]  [61, 41, 17, 2]
    euclidean on sqrt(p)  [58, 42, 17, 5, 2] x3   (identical)

(cluster counts per PLSCAN layer). Since layers are selected **positionally**
(``ensemble[2]``), an unstable layer *count* silently changes what that index
means — one earlier ``jensen_shannon`` run produced only three layers, where
``[2]`` would have been 2 meta-LCs instead of 17. The exact path removes that.

Note on the ``Reducer`` pattern (UMAP 50-D cosine graph + ``lensed_umap``
repulsion ramp -> PLSCAN euclidean): that exists to work around the standalone
``plscan`` package validating ``metric`` against ``KDTree.valid_metrics``, which
excludes cosine. ``fast_hdbscan.PLSCAN`` has a pynndescent branch and no such
restriction, and at 2700 topics the cost it optimises is not the bottleneck — so
we take the geometry directly rather than through a stochastic embedding.
"""

import numpy as np
import pandas as pd

_MIN_SAMPLES = 5


def hellinger_embed(topic_feature_df, copy=True):
    """Map topic distributions to the space where euclidean == Hellinger.

    Rows are L1-renormalised first: ``sqrt(p)`` only means Hellinger if ``p`` is
    a distribution. ``run_nmf_ensemble`` already guarantees this
    (``l1_normalize=True``), but LDA topics arrive by a different route, so the
    normalisation is done here rather than assumed.

    Parameters
    ----------
    topic_feature_df : pd.DataFrame or ndarray, shape (n_topics, n_features)
        Non-negative topic x feature matrix.
    copy : bool
        Ignored for DataFrames (always materialised); kept for call-site clarity.

    Returns
    -------
    ndarray (n_topics, n_features), float64 — ``sqrt(p)``, rows of unit L2 norm.
    """
    X = topic_feature_df.values if isinstance(topic_feature_df, pd.DataFrame) \
        else np.asarray(topic_feature_df)
    X = np.asarray(X, dtype=np.float64)
    if np.any(X < 0):
        raise ValueError(
            "topic vectors must be non-negative to be read as distributions."
        )
    s = X.sum(axis=1, keepdims=True)
    zero = (s[:, 0] == 0)
    if zero.any():
        # an all-zero topic has no distribution; leave it at the origin rather
        # than inventing a uniform one, and let PLSCAN call it noise
        s = s.copy()
        s[zero] = 1.0
    return np.sqrt(X / s)


def embed_topics(topic_feature_df, mode="hellinger", n_components=10,
                 random_state=42, **umap_kwargs):
    """The array the clustering runs on. Two modes, both rooted in Hellinger.

    ``"hellinger"`` (default)
        ``sqrt(p)`` itself — euclidean there *is* the Hellinger distance, so
        PLSCAN clusters the topic distributions directly. Deterministic, no
        parameters, and the distances are exactly the ones the profiles are
        built from.

    ``"umap"``
        A ``n_components``-D UMAP **of that same array**, clustered with
        euclidean — the only metric a UMAP output has, since the optimisation
        places points so that output euclidean distance approximates the input's
        fuzzy topology. At 10 dimensions PLSCAN still gets its exact
        KDTree/Boruvka MST rather than an approximate kNN graph.

    The trade. UMAP is deterministic once ``random_state`` is fixed, so this
    does *not* reintroduce run-to-run randomness. What it does introduce is a
    dependence of the meta-LC ids on the UMAP seed and version, where the
    ``"hellinger"`` path depends on neither. It also puts a fitted, lossy step
    between the topics and the clusters: the clusters are then structure in the
    embedding, which is a hypothesis about the topics rather than the topics.

    Pairing with the 2-D picture. :func:`project_topics` embeds the *same*
    ``sqrt(p)`` array; with the same ``random_state`` and ``n_neighbors`` the two
    UMAPs share their **kNN graph**, since graph construction does not depend on
    ``n_components``. They do **not** share a layout — each is its own
    optimisation, so the 2-D is not a projection of the 10-D and points can still
    overlap across cluster colours. The shared graph is a real common footing,
    not a guarantee.

    Returns
    -------
    ndarray (n_topics, n_features) for ``"hellinger"``, else
    (n_topics, n_components).
    """
    X = hellinger_embed(topic_feature_df)
    if mode == "hellinger":
        return X
    if mode != "umap":
        raise ValueError(f'mode must be "hellinger" or "umap", got {mode!r}')

    import umap

    return umap.UMAP(n_components=n_components, metric="euclidean",
                     random_state=random_state, **umap_kwargs).fit_transform(X)


def cluster_topics(topic_feature_df, min_samples=_MIN_SAMPLES, layer=None,
                   reproducible=True, **plscan_kwargs):
    """PLSCAN the pooled topics in Hellinger geometry.

    Drop-in for ``archive.run_plscan_features`` — same 4-tuple, same meaning —
    but clusters ``sqrt(p)`` under euclidean, so PLSCAN uses its exact
    KDTree/Boruvka MST instead of an approximate kNN graph.

    Parameters
    ----------
    topic_feature_df : pd.DataFrame or ndarray
        Pooled topics, one row per topic.
    min_samples : int
        PLSCAN ``min_samples``.
    layer : int, optional
        Which PLSCAN layer to return labels for. ``None`` picks the
        highest-persistence layer.
    reproducible : bool
        Passed to PLSCAN.
    **plscan_kwargs
        Anything else PLSCAN accepts (``max_layers``, ``base_min_cluster_size``,
        ``cannot_link``, ...). ``metric`` is fixed by the geometry and rejected.

    Returns
    -------
    (clusterer, labels, n_clusters, chosen_layer)
        ``clusterer`` is the fitted PLSCAN — it carries every layer, so callers
        can re-index without refitting. ``labels`` has ``-1`` for noise.
    """
    from fast_hdbscan import PLSCAN

    if "metric" in plscan_kwargs:
        raise TypeError(
            "cluster_topics fixes metric='euclidean' on the Hellinger embedding; "
            "call fast_hdbscan.PLSCAN directly to cluster in another geometry."
        )

    X = hellinger_embed(topic_feature_df)
    clusterer = PLSCAN(min_samples=min_samples, metric="euclidean",
                       reproducible=reproducible, **plscan_kwargs)
    clusterer.fit(X)

    chosen_layer = (int(np.argmax(clusterer.layer_persistence_scores_))
                    if layer is None else int(layer))
    labels = np.asarray(clusterer.cluster_layers_[chosen_layer])
    n_clusters = len(set(labels.tolist()) - {-1})
    return clusterer, labels, n_clusters, chosen_layer


def _canonical(labels):
    """Relabel a partition by order of first appearance, keeping ``-1``.

    Cluster ids are arbitrary, so two runs can produce the *same* partition under
    permuted ids. Voting on raw ids would count those as disagreeing. This maps
    any partition to one canonical form so identical partitions compare equal.
    """
    labels = np.asarray(labels)
    out = np.full(labels.shape, -1, dtype=np.int64)
    nxt = 0
    seen = {}
    for i, v in enumerate(labels.tolist()):
        if v < 0:
            continue
        if v not in seen:
            seen[v] = nxt
            nxt += 1
        out[i] = seen[v]
    return tuple(out.tolist())


def cluster_topics_stable(topic_feature_df, n_repeats=10, min_samples=_MIN_SAMPLES,
                          layer=None, reproducible=True, X=None, **plscan_kwargs):
    """:func:`cluster_topics` repeated ``n_repeats`` times, returning the mode.

    ``PLSCAN`` is **not reproducible on this data, even inside one process.**
    Measured on the HRD LDA ensemble (2,700 topics x 6,866 ASVs), six repeats of
    the identical call on the identical embedding gave five identical labellings
    and one that differed (layer-1 noise 1,733 vs 1,735). Across processes it is
    worse: one run returned layers ``[43, 22, 5]`` where every other run returned
    ``[43, 15, 3]``. Since layers are selected **positionally**, that silently
    changes what ``cluster_layers_[1]`` means, and with it every meta-LC id the
    analyst has been reasoning about.

    This runs the fit ``n_repeats`` times on **one** embedding (the embedding is
    pure numpy and deterministic, so it is not the source of the variation) and
    returns the most frequent result. Runs are compared as partitions under
    :func:`_canonical`, so a relabelling of the same partition counts as
    agreement rather than as a third opinion.

    The winner is returned with its **own native cluster ids**, not renumbered —
    so the modal run's ids are the ids you get, and they can be persisted.

    Returns
    -------
    (clusterer, labels, n_clusters, chosen_layer, votes)
        The first four match :func:`cluster_topics`. ``votes`` is a list of
        dicts, most-frequent first, each with ``votes``, ``layer_sizes``,
        ``noise`` and ``ari_to_mode``.

        **Read ``ari_to_mode``, not ``layer_sizes``.** On HRD, ten repeats gave
        three distinct partitions that *all* had layer sizes ``[43, 15, 3]`` —
        the counts agree while the membership does not. A vote split with high
        ARI (>0.99) is a few topics moving between clusters and is harmless; a
        low ARI means the meta-LCs themselves are not determined by the data,
        and no amount of voting will fix that.
    """
    from fast_hdbscan import PLSCAN

    if "metric" in plscan_kwargs:
        raise TypeError(
            "cluster_topics_stable fixes metric='euclidean' on the Hellinger "
            "embedding; call fast_hdbscan.PLSCAN directly for another geometry."
        )

    # `X` lets the caller pass the space to cluster in -- either an embedding
    # already computed (so it is not rebuilt for the medoids and the plot) or a
    # different one from `embed_topics`, e.g. the 10-D UMAP. Default keeps the
    # Hellinger array.
    X = hellinger_embed(topic_feature_df) if X is None else np.asarray(X)

    def fit_once():
        c = PLSCAN(min_samples=min_samples, metric="euclidean",
                   reproducible=reproducible, **plscan_kwargs)
        c.fit(X)
        return c

    clusterer, votes = vote_clusterers(fit_once, n_repeats=n_repeats)

    chosen_layer = (int(np.argmax(clusterer.layer_persistence_scores_))
                    if layer is None else int(layer))
    labels = np.asarray(clusterer.cluster_layers_[chosen_layer])
    n_clusters = len(set(labels.tolist()) - {-1})
    return clusterer, labels, n_clusters, chosen_layer, votes


def vote_clusterers(fit_once, n_repeats=10):
    """Fit ``n_repeats`` times and return the modal clustering plus a vote table.

    ``fit_once`` is a zero-argument callable returning a **fitted** clusterer
    exposing ``cluster_layers_``. Keeping the geometry *outside* this function is
    the whole point: the same vote then serves the Hellinger path
    (:func:`cluster_topics_stable`), the legacy JSD paths in ``archive``, and
    the notebooks that PLSCAN a 2-D UMAP embedding directly — none of which
    share an embedding, but all of which share the instability.

    Runs are compared as partitions under :func:`_canonical`, so a relabelling of
    the same partition counts as agreement rather than as a rival answer. The
    winner is returned with its **own native cluster ids**, unrenumbered, so the
    modal run's ids are the ids the caller gets and can persist.

    Returns
    -------
    (clusterer, votes)
        ``votes`` is a list of dicts, most-frequent first, each with ``votes``,
        ``layer_sizes``, ``noise`` and ``ari_to_mode``. **Read ``ari_to_mode``,
        not ``layer_sizes``** — distinct partitions can share the same sizes.
    """
    from collections import Counter
    from sklearn.metrics import adjusted_rand_score

    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")

    keys, reps = [], {}
    for _ in range(n_repeats):
        c = fit_once()
        key = tuple(_canonical(l) for l in c.cluster_layers_)
        keys.append(key)
        reps.setdefault(key, c)

    counts = Counter(keys)
    winner, _ = counts.most_common(1)[0]
    clusterer = reps[winner]

    votes = []
    for key, n in counts.most_common():
        rep = reps[key]
        votes.append({
            "votes": n,
            "layer_sizes": [len(set(l) - {-1}) for l in key],
            "noise": [sum(1 for v in l if v < 0) for l in key],
            "ari_to_mode": [
                round(float(adjusted_rand_score(
                    np.asarray(clusterer.cluster_layers_[i]),
                    np.asarray(rep.cluster_layers_[i]))), 4)
                if i < len(clusterer.cluster_layers_) else None
                for i in range(len(key))
            ],
        })
    return clusterer, votes


def cluster_medoids(topic_feature_df, labels, probabilities=None, X=None):
    """Cluster medoids in the same space PLSCAN clustered in.

    Euclidean on ``sqrt(p)`` — i.e. the Hellinger medoid — so the medoid is
    chosen under the distance that formed the cluster. It is still a **real
    topic row**, so downstream profiles remain raw topic vectors and stay
    interpretable.

    Returns
    -------
    list[int] — global row indices, ordered by cluster id.
    """
    import ensemble as E

    # Must be the space the clustering ran in, so pass the same `X` when the
    # clustering used something other than the default Hellinger array --
    # otherwise the medoid is picked under a distance that did not form the
    # cluster. The returned indices are row numbers either way, so the medoid
    # stays a real topic.
    X = hellinger_embed(topic_feature_df) if X is None else np.asarray(X)
    return E.compute_medoids_from_features(
        pd.DataFrame(X), labels, probabilities=probabilities, metric="euclidean",
    )


def project_topics(topic_feature_df, n_components=2, random_state=42, X=None,
                   **umap_kwargs):
    """2-D overview of the pooled topics, in **the clustering's own geometry**.

    Same two steps as :func:`cluster_topics` — embed with :func:`hellinger_embed`,
    then treat it as euclidean — only the last step differs: UMAP down to 2-D
    instead of PLSCAN. So the scatter is a projection of exactly the space the
    clusters were formed in, not merely of a distance that resembles it.

    This matters because the figure colours points by cluster. If the projection
    used a different geometry, points could legitimately land inside the wrong
    cluster's colour and the picture would misrepresent the clustering. Nothing
    here feeds back into the clustering — it is display only.

    Parameters
    ----------
    topic_feature_df : pd.DataFrame or ndarray
        The same pooled topics passed to :func:`cluster_topics`.
    n_components : int
        Output dimensions (2 for the standard overview).
    random_state : int
        UMAP seed. UMAP's layout is stochastic; the clustering is not, so a
        different seed moves the points without moving the cluster assignments.
    **umap_kwargs
        Forwarded to ``umap.UMAP`` (``n_neighbors``, ``min_dist``, ...).
        ``metric`` is fixed by the geometry and rejected.

    Returns
    -------
    ndarray (n_topics, n_components)
    """
    import umap

    if "metric" in umap_kwargs:
        raise TypeError(
            "project_topics fixes metric='euclidean' on the Hellinger embedding "
            "so the picture matches the clustering; use archive.run_umap_features "
            "for an overview in some other geometry."
        )

    # Always the sqrt(p) array, never a clustering embedding: with the same
    # random_state and n_neighbors this UMAP and a `embed_topics(mode="umap")`
    # one share their kNN graph, which is the point of pairing them.
    X = hellinger_embed(topic_feature_df) if X is None else np.asarray(X)
    reducer = umap.UMAP(n_components=n_components, metric="euclidean",
                        random_state=random_state, **umap_kwargs)
    return reducer.fit_transform(X)


def layer_table(clusterer, selected=None):
    """Summary of every PLSCAN layer: n_clusters / n_noise / persistence.

    Mirrors ``Ensemble.layer()`` but works on a bare clusterer, so the LDA track
    can inspect layers without building an ``Ensemble``.
    """
    rows = []
    for i, (lab, p) in enumerate(zip(clusterer.cluster_layers_,
                                     clusterer.layer_persistence_scores_)):
        lab = np.asarray(lab)
        rows.append({
            "layer": i,
            "n_clusters": len({c for c in lab.tolist() if c >= 0}),
            "n_noise": int((lab == -1).sum()),
            "persistence": round(float(p), 3),
            "selected": i == selected,
        })
    return pd.DataFrame(rows)
