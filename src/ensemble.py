"""Method-agnostic ensemble orchestration — shared by the LDA and NMF tracks.

    ensemble = run_ensemble(df, [2, 10], method="lda")   # fit + cluster
    ensemble.layer()            # table of every PLSCAN layer; object defaults to best
    ensemble[3]                 # same ensemble, but using layer 3

    pp = ProfilePlot(ensemble)
    pp.top_taxa(profile="medoid", order_by="first_k")     # profile small-multiples
    pp.projection(profile="medoid")                        # UMAP overview

    stripesankey(ensemble, layer=2)   # meta-LC composition across k (plotly)

Everything in here — fitting orchestration, caching, clustering glue, profile
building — treats ``method`` as a parameter, not a branch point baked into the
module. The two things that genuinely differ between LDA and NMF — the
ensemble fitter itself, and the fixed-W inference math — live in ``lda.py`` /
``nmf.py`` and are called from here by name. The shared clustering
**geometry** (Hellinger embedding, PLSCAN, medoids, the 2-D projection) lives
in ``clustering.py``. Actual drawing lives in ``plotting.py`` — this
file calls into it (``ProfilePlot`` / ``stripesankey``) but contains no
``matplotlib``/``plotly`` code itself. Cluster labels are ``"MetaLC-<id>"``.
"""

import copy
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

import lda
import nmf
import plotting
from clustering import (cluster_topics_stable, cluster_medoids,
                              embed_topics)

# Hard-coded pipeline settings (per the design note).
_N_RUNS = 50
_MIN_SAMPLES = 5
# The geometry lives in clustering (shared with the LDA track): sqrt(p),
# then euclidean == Hellinger. `cluster_topics` ends it with PLSCAN (exact
# KDTree MST); `project_topics` ends the same two steps with a 2-D UMAP, so the
# overview scatter and the clustering cannot drift apart.


def _as_krange(k_range):
    """Accept [lo, hi] (inclusive, as in the note) or any iterable of K values."""
    if isinstance(k_range, (list, tuple)) and len(k_range) == 2:
        lo, hi = int(k_range[0]), int(k_range[1])
        return range(lo, hi + 1)
    return k_range


class Ensemble:
    """Result of :func:`run_ensemble` — the pooled topics clustered by PLSCAN
    across all resolution layers. Defaults to the highest-persistence (best) layer;
    index it (``ensemble[i]``) to work at layer ``i`` instead.
    """

    def __init__(self, df, topic_feature_df, topic_meta_df, clusterer,
                 group_map=None, n_runs=_N_RUNS, votes=None, method="nmf"):
        self.df = df                              # samples x features (V)
        self.sample_ids = list(df.index)
        self.feature_names = list(df.columns)
        self.topic_feature_df = topic_feature_df  # topics x features (pooled)
        # k / run / topic / weight, plus that fit's convergence record
        # (see lda.run_lda_ensemble / nmf.run_nmf_ensemble). Consumers index
        # this frame positionally or by name, so the extra columns ride along
        # harmlessly.
        self.topic_meta_df = topic_meta_df
        self.clusterer = clusterer                # fitted PLSCAN (all layers)
        self.group_map = group_map                # feature -> group (e.g. ASV->genus)
        self.n_runs = n_runs
        # Which fitter produced these topics. Carried on the object rather than
        # passed at call time because inference must dispatch on it: an
        # LDA ensemble folded in under NMF-GKL would silently lose the Dirichlet
        # alpha prior that regularises H against the redundant medoid dictionary.
        self.method = method
        # PLSCAN is not reproducible (see clustering.cluster_topics_stable),
        # so `clusterer` is the *mode* of several fits. `votes` records how much
        # the fits disagreed — check `ari_to_mode` before trusting a meta-LC count.
        self.votes = votes
        self.best_layer = int(np.argmax(clusterer.layer_persistence_scores_))
        self._layer = self.best_layer

    # -- layer selection ---------------------------------------------------
    @property
    def selected_layer(self):
        return self._layer

    def __getitem__(self, i):
        """``ensemble[i]`` -> the same ensemble viewing layer ``i``."""
        n = len(self.clusterer.cluster_layers_)
        if not 0 <= int(i) < n:
            raise IndexError(f"layer {i} out of range (0..{n - 1})")
        new = copy.copy(self)
        new._layer = int(i)
        return new

    @property
    def labels(self):
        """Integer cluster label per pooled topic for the selected layer (-1 noise)."""
        return np.asarray(self.clusterer.cluster_layers_[self._layer])

    @property
    def membership(self):
        return self.clusterer.membership_strengths_

    @property
    def n_clusters(self):
        return len(set(self.labels.tolist()) - {-1})

    def layers(self):
        """Summary table of every PLSCAN layer (n_clusters, n_noise, persistence).

        The selected layer (best by default) is flagged in the ``selected`` column.
        Index the ensemble (``ensemble[i]``) to move to another one.
        """
        rows = []
        for i, (lab, p) in enumerate(zip(
            self.clusterer.cluster_layers_, self.clusterer.layer_persistence_scores_,
        )):
            lab = np.asarray(lab)
            rows.append({
                "layer": i,
                "n_clusters": len({c for c in lab.tolist() if c >= 0}),
                "n_noise": int((lab == -1).sum()),
                "persistence": round(float(p), 3),
                "selected": i == self._layer,
            })
        return pd.DataFrame(rows)

    # `layer()` was the original name; kept so existing notebooks keep working.
    layer = layers

    def __repr__(self):
        return (f"Ensemble({self.method.upper()}, {len(self.topic_feature_df)} topics, "
                f"{len(self.clusterer.cluster_layers_)} layers, "
                f"layer={self._layer} [best={self.best_layer}], "
                f"{self.n_clusters} meta-LCs)")


def fit_budget(method, fit_params=None):
    """The convergence budget a fit of ``method`` runs under, for the manifest.

    Pooled topics are reused on sight, so this is what makes a change to the
    budget visible: ``metalc._check_fit_budget`` compares it against what the
    cache was written with and refuses a mismatch rather than answering with
    topics fitted under rules nobody is using any more.
    """
    if method == "lda":
        budget = {"passes": None, "max_passes": lda.MAX_PASSES,
                  "rel_tol": lda.REL_TOL, "iterations": lda.ITERATIONS,
                  "move_tol": None}
    else:
        budget = {"max_iter": nmf.MAX_ITER, "tol": 1e-4, "init": "random"}
    budget.update({k: v for k, v in (fit_params or {}).items() if k in budget})
    # The window only means something under the window rule. Recording it
    # unconditionally would make every manifest written before it existed
    # (HRD lda_ensemble_refit among them) read as a changed stopping rule.
    if budget.get("move_tol") is not None:
        budget["move_window"] = (fit_params or {}).get("move_window",
                                                       lda.MOVE_WINDOW)
    return budget


#: Budget keys that are **ceilings**: raising one can only let a fit run longer,
#: so a cache written under a lower ceiling is still valid as long as nothing
#: actually hit it. Every other key defines the stopping *rule*, and changing
#: one changes the answer.
CEILING_KEYS = {"max_passes", "max_iter"}


def legacy_fit_budget(method):
    """What a cache with no recorded budget was fitted under.

    Manifests written before convergence tracking have no ``fit`` key. They are
    not unknown, though — they all came from the same two hard-coded defaults.
    """
    if method == "lda":
        return {"passes": 100, "max_passes": 100, "rel_tol": None,
                "iterations": 400}
    return {"max_iter": 500, "tol": 1e-4, "init": "random"}


def run_ensemble(df, k_range, method="nmf", group_map=None, cache_dir=None,
                 n_repeats=10, workers=1, n_runs=_N_RUNS, embed="hellinger",
                 umap_random_state=42, embed_dim=10, clustering=None,
                 fit_params=None):
    """Fit an ensemble and cluster the pooled topics with PLSCAN.

    Parameters
    ----------
    df : pd.DataFrame
        Count matrix, rows = samples, columns = features (e.g. ASVs). Both
        fitters take the *same* matrix — feed them different ones and their
        meta-LCs stop being comparable. Counts are the standard input: LDA's
        multinomial model assumes them and GKL is the matching NMF loss.
    k_range : [lo, hi] or iterable
        Numbers of components to sweep. ``[2, 10]`` means 2..10 inclusive.
    method : {"nmf", "lda"}
        Which fitter produces the pooled topics. ``"nmf"`` is sklearn NMF under
        generalized-KL; ``"lda"`` is gensim variational LDA
        (:func:`lda.run_lda_ensemble`). Both then go through the
        same Hellinger geometry, so their meta-LCs are formed identically. The
        choice is recorded on the returned ``Ensemble`` and drives fixed-W
        inference.
    workers : int
        Parallel worker processes for the fitting step (``method="lda"`` only;
        sklearn's NMF does its own threading).
    n_runs : int
        Fits per k. 50 is the pipeline default; lower it only for smoke tests —
        the ensemble's whole premise is that many seeds explore the objective.
    embed : {"hellinger", "umap"}
        The space PLSCAN clusters in, via :func:`clustering.embed_topics`.
        ``"hellinger"`` (default) is ``sqrt(p)`` itself; ``"umap"`` is an
        ``embed_dim``-D UMAP of it. Read that function on what the UMAP path
        costs you.
    embed_dim, umap_random_state : int
        Only used by ``embed="umap"``. The same ``umap_random_state`` is what the
        2-D overview uses, so the two UMAPs share a kNN graph.
    group_map : dict, optional
        Feature -> group name (e.g. ASV -> genus) used only to make profiles
        readable (see :class:`ProfilePlot`). Fitting stays in the feature space.
    cache_dir : str, optional
        If given, the pooled topics are cached there (``all_topics_feature.npy`` +
        metadata; the older CSV layout is still read) and reused on the next call
        — so the expensive fitting runs once. PLSCAN is always (re)run; it is
        cheap.
    n_repeats : int
        How many PLSCAN fits to vote over. PLSCAN is **not reproducible**, not
        even within one process, and its layer *count* can change between runs —
        which silently re-points a positional ``ensemble[2]``. The clustering is
        therefore the **mode** of ``n_repeats`` fits, not a single draw. Set to 1
        for the old single-fit behaviour. The vote is on ``Ensemble.votes``.

    Returns
    -------
    Ensemble  (defaults to the best PLSCAN layer).
    """
    feature_names = list(df.columns)
    cached = cache_dir and any(
        os.path.exists(os.path.join(cache_dir, f))
        for f in ("all_topics_feature.npy", "all_topics_feature.csv")
    )
    if method not in ("nmf", "lda"):
        raise ValueError(f'method must be "nmf" or "lda", got {method!r}')

    if cached:
        topic_feature_df, topic_meta_df, _ = load_ensemble(cache_dir)
    elif method == "lda":
        topic_feature_df, topic_meta_df = lda.run_lda_ensemble(
            df.values, feature_names, k_range=_as_krange(k_range),
            n_runs=n_runs, workers=workers, **(fit_params or {}),
        )
    else:
        topic_feature_df, topic_meta_df = nmf.run_nmf_ensemble(
            df.values.astype(float), feature_names,
            k_range=_as_krange(k_range), n_runs=n_runs,
            **{"max_iter": nmf.MAX_ITER, **(fit_params or {})},
        )
    if not cached and cache_dir:
        save_ensemble(cache_dir, topic_feature_df, topic_meta_df, jsd_dist=None)

    if clustering is not None:
        # Restored from cache -- see metalc.run_ensemble. Skipping the vote is
        # what keeps meta-LC ids identical across sessions; PLSCAN is not
        # reproducible, so re-running it can renumber (or re-count) the layers
        # an analyst has already been reasoning about.
        clusterer, votes, X = clustering
        if X is None:
            X = embed_topics(topic_feature_df, mode=embed,
                             n_components=embed_dim, random_state=umap_random_state)
    else:
        # Built once and kept on the Ensemble: the clustering, the medoids and
        # the 2-D picture all read it, instead of each recomputing sqrt(p)
        # (~2s a time on 2700x6866).
        X = embed_topics(topic_feature_df, mode=embed, n_components=embed_dim,
                         random_state=umap_random_state)
        clusterer, _labels, _n, _layer, votes = cluster_topics_stable(
            topic_feature_df, n_repeats=n_repeats, min_samples=_MIN_SAMPLES, X=X,
        )
    ens = Ensemble(df, topic_feature_df, topic_meta_df, clusterer,
                   group_map=group_map, n_runs=n_runs, votes=votes,
                   method=method)
    ens.embedding = X
    ens.embed_mode = embed
    ens.umap_random_state = umap_random_state
    return ens


class ProfilePlot:
    """Profile views for an :class:`Ensemble`'s selected layer.

    Builds the medoid and average meta-LC profiles once; ``top_taxa`` and
    ``projection`` then render either (via ``plotting.py``). When the ensemble
    carries a ``group_map`` the profiles are aggregated to that grouping (e.g.
    genus) for readability.
    """

    def __init__(self, ensemble, group_map=None):
        self.e = ensemble
        # Overrides ensemble.group_map for this instance. `metalc.plot_meta`
        # uses it to pick the taxonomic rank at plot time, since the rank is a
        # display choice and must not be baked in when the ensemble is fitted.
        self.group_map = group_map if group_map is not None else ensemble.group_map
        self.labels = ensemble.labels
        self.n_clusters = ensemble.n_clusters
        self.medoid_indices = cluster_medoids(
            ensemble.topic_feature_df, self.labels,
            probabilities=ensemble.membership,
            X=getattr(ensemble, "embedding", None),
        )
        self._feature_profile = {
            kind: build_profiles(
                ensemble.topic_feature_df, self.labels, self.medoid_indices,
                self.n_clusters, kind=kind,
            )
            for kind in ("medoid", "average")
        }

    def _display(self, profile):
        """The profile as shown: aggregated to group_map (genus) if present."""
        p = self._feature_profile[profile]
        if self.group_map is not None:
            return aggregate_features(p, self.group_map, drop_groups=["Undefined"])
        return p

    def top_taxa(self, profile="medoid", order_by="first_k", feature_cutoff=0.05,
                 save_path=None, title=None, row_annotations=None):
        """Small-multiples of each meta-LC's top features (medoid or average),
        ordered by K-emergence (``"first_k"``) or by JSD clustering (``"clustering"``).

        ``title`` defaults to naming NMF-GKL. An :class:`Ensemble` built from
        pooled LDA topics (this class is shared by both tracks) should pass its
        own.
        """
        return plotting.plot_top_taxa(
            self._display(profile), self.labels, self.e.topic_meta_df,
            order_by=order_by, feature_cutoff=feature_cutoff, n_runs=self.e.n_runs,
            title=title or f"NMF-GKL meta-LC profiles — {profile.upper()}",
            save_path=save_path, row_annotations=row_annotations,
        )

    def projection(self, profile="medoid", feature_cutoff=0.05, save_path=None,
                   umap_random_state=42, title=None, sub_title=None):
        """UMAP overview of the pooled topics, coloured by PLSCAN cluster and
        labelled by each meta-LC's top features (from ``profile``).

        Projected through ``clustering.project_topics`` — the same
        ``sqrt(p)`` + euclidean geometry :func:`clustering.cluster_topics`
        clusters in, with UMAP in place of PLSCAN as the final step. The
        picture and the clusters therefore cannot drift apart.
        ``umap_random_state`` moves the layout only; the clustering is
        deterministic and unaffected.
        """
        kw = {k: v for k, v in (("title", title), ("sub_title", sub_title))
              if v is not None}
        return plotting.plot_projection(
            self.e.topic_feature_df, self.labels, self._display(profile),
            self.n_clusters, umap_random_state=umap_random_state,
            feature_cutoff=feature_cutoff, save_path=save_path, **kw,
        )


def stripesankey(ensemble, layer=None, include_noise=False, normalize=False,
                 node_labels=None, save_path=None, **kwargs):
    """Meta-LC composition across k, as a stripe + Sankey (plotly ``Figure``).

    One column per ``k``, one rectangle per meta-LC, **height = the number of
    pooled topics fitted at that k that PLSCAN put in that meta-LC**. The point
    is that ``k`` does not fix the number of meta-LCs: the ``n_runs`` fits at
    ``k=2`` are independent, so their topics can spread over 3 meta-LCs. Thin
    constant-width connectors link a meta-LC to itself in the next column.

    Deliberately a module-level function rather than a :class:`ProfilePlot`
    method — ``ProfilePlot.__init__`` eagerly computes medoids and both profile
    kinds, and this figure needs none of that.

    Parameters
    ----------
    ensemble : Ensemble
    layer : int, optional
        PLSCAN layer to read the meta-LCs from. ``None`` (default) keeps the
        ensemble's current layer; otherwise ``ensemble[layer]``. Call
        ``ensemble.layer()`` for the table of layers to choose from.
    include_noise : bool
        Show PLSCAN's unassigned (``-1``) topics as a grey ``noise`` node. False
        by default — on HRD the noise node dwarfs the real meta-LCs (320 of the
        500 topics at k=10) and squashes everything else flat. Dropping it only
        removes the *rectangle*: the shares and the per-column caption stay over
        the full ``n_runs * k`` topics fitted, so each column still reports how
        many of them were assigned.
    normalize : bool
        Level the columns so each k reads as shares rather than raw counts.
    node_labels : dict, optional
        ``{meta_lc_id: str}`` drawn on the nodes (e.g. each meta-LC's top genus).
    save_path : str, optional
        ``.html`` (interactive) or ``.png``/``.pdf``/``.svg`` (needs kaleido).
    **kwargs
        Passed through to :func:`plotting.plot_meta_stripesankey`
        (``link_width``, ``gap_frac``, ``palette``, ``width``, ``height``, ...).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    e = ensemble if layer is None else ensemble[layer]
    # always count noise first, so the per-k denominator stays the true n_runs*k
    counts = meta_lc_k_counts(e.labels, e.topic_meta_df, include_noise=True)
    runs = meta_lc_k_counts(e.labels, e.topic_meta_df, include_noise=True,
                            value="runs")
    totals = counts.sum(axis=1)
    if not include_noise:
        counts = counts.drop(columns=[NOISE_KEY], errors="ignore")
        runs = runs.drop(columns=[NOISE_KEY], errors="ignore")
    return plotting.plot_meta_stripesankey(
        counts, runs=runs, column_totals=totals, normalize=normalize,
        node_labels=node_labels,
        title=f"Meta-LC composition across k — PLSCAN layer {e.selected_layer}",
        save_path=save_path, **kwargs,
    )


# --------------------------------------------------------------------------- #
# Per meta-cluster ("meta-LC") profiles: medoid or average
# --------------------------------------------------------------------------- #
def build_profiles(
    topic_feature_df,
    labels,
    medoid_indices,
    n_clusters,
    kind="medoid",
    index_prefix="MetaLC",
):
    """Per-cluster feature profile — the medoid topic or the cluster average.

    Both are computed from the same pooled, L1-normalised topics, so they are on
    the same (probability) scale and directly comparable in a plot.

    Parameters
    ----------
    topic_feature_df : pd.DataFrame (n_topics, n_features)
        The pooled topics (rows) from ``run_ensemble``.
    labels : array-like (n_topics,)
        Cluster id per topic; ``-1`` (noise) is ignored for the average.
    medoid_indices : list[int]
        Global row indices of the cluster medoids, ordered by cluster id
        (as returned by ``clustering.cluster_medoids``).
    n_clusters : int
        Number of clusters (ids ``0..n_clusters-1``).
    kind : {"medoid", "average"}
        ``"medoid"`` -> the medoid topic's vector.
        ``"average"`` -> mean of all member topics' vectors, renormalised to sum 1
        (the "meta-LC average profile").

    Returns
    -------
    pd.DataFrame, rows = ``f"{index_prefix}-{i}"``, cols = features.
    """
    labels = np.asarray(labels)

    if kind == "medoid":
        if len(medoid_indices) != n_clusters:
            raise ValueError(
                f"medoid_indices has {len(medoid_indices)} entries, "
                f"expected n_clusters={n_clusters}"
            )
        df = topic_feature_df.iloc[list(medoid_indices)].copy().reset_index(drop=True)

    elif kind == "average":
        vals = topic_feature_df.values
        prof = np.zeros((n_clusters, vals.shape[1]), dtype=float)
        for c in range(n_clusters):
            members = vals[labels == c]
            if members.shape[0] == 0:
                continue
            m = members.mean(axis=0)
            s = m.sum()
            prof[c] = m / s if s > 0 else m
        df = pd.DataFrame(prof, columns=topic_feature_df.columns)

    else:
        raise ValueError(f"kind must be 'medoid' or 'average', got {kind!r}")

    df.index = [f"{index_prefix}-{i}" for i in range(n_clusters)]
    return df


def aggregate_features(
    profile_df,
    group_map,
    drop_groups=None,
    renormalize=True,
    sort=True,
    unknown="Unknown",
):
    """Aggregate a fine-feature profile to a coarser grouping (e.g. ASV -> genus).

    Fitting at ASV level keeps the topics/W in ASV space (needed for fixed-W
    inference against the ASV count table), but ASV profiles are unreadable. This
    collapses each profile's columns by ``group_map`` (summing member ASVs into
    their genus), optionally drops buckets like ``'Undefined'`` and renormalises
    each row back to a distribution, and optionally sorts columns by descending
    total weight.

    Parameters
    ----------
    profile_df : pd.DataFrame
        Rows = meta-LCs, cols = fine features (e.g. ASV ids).
    group_map : dict or pandas Series
        Fine-feature -> group name (e.g. ASV id -> genus).
    drop_groups : list[str] or None
        Groups to drop after aggregation (e.g. ``['Undefined']``).
    renormalize : bool
        Rescale each row to sum 1 after dropping.
    sort : bool
        Order columns by descending total weight (like the LDA genus profile).

    Returns
    -------
    pd.DataFrame  rows = meta-LCs (unchanged index), cols = groups.
    """
    gm = dict(group_map) if not isinstance(group_map, dict) else group_map
    groups = [gm.get(c, unknown) for c in profile_df.columns]

    g = profile_df.copy()
    g.columns = groups
    g = g.T.groupby(level=0).sum().T

    if drop_groups:
        g = g.drop(columns=[x for x in drop_groups if x in g.columns], errors="ignore")
    if renormalize:
        rs = g.sum(axis=1)
        rs[rs == 0] = 1.0
        g = g.div(rs, axis=0)
    if sort:
        g = g[g.sum().sort_values(ascending=False).index]
    return g


def build_cluster_labels(profile_df, n_clusters, labels, feature_cutoff=0.1,
                         index_prefix="MetaLC"):
    """datamapplot per-point labels.

    Uses a non-breaking-space trick so datamapplot keeps each segment
    ("MetaLC-3", "Bacteroides (11%)", ...) on its own line.

    Returns (point_labels, highlight_label_list).
    """
    NBSP = " "

    def _nbsp(s):
        return s.replace(" ", NBSP)

    label_of = {}
    highlight = []
    for c in range(n_clusters):
        profile = profile_df.iloc[c]
        top = profile[profile > feature_cutoff].sort_values(ascending=False)
        if len(top) == 0:
            top = profile.sort_values(ascending=False).head(1)
        segments = [_nbsp(f"{name} ({val:.0%})") for name, val in top.items()]
        lbl = " ".join([_nbsp(f"{index_prefix}-{c}")] + segments)
        label_of[c] = lbl
        highlight.append(lbl)

    labels = np.asarray(labels)
    point_labels = np.array([
        label_of.get(int(l), "Unlabelled") if l != -1 else "Unlabelled"
        for l in labels
    ])
    return point_labels, highlight


def lc_cluster_order(labels, topic_meta_df, first_k_map, stable_k_map):
    """Order clusters by (first_k, peak_k, stable_k). Thin pass-through to
    :func:`plotting.k_dist_cluster_order` — kept here because callers import
    the ordering step alongside the other orchestration in this module.
    """
    return plotting.k_dist_cluster_order(labels, topic_meta_df, first_k_map, stable_k_map)


# --------------------------------------------------------------------------- #
# Cache the pooled ensemble
# --------------------------------------------------------------------------- #
def save_ensemble(out_dir, topic_feature_df, topic_meta_df, jsd_dist=None,
                  doc_topic=None, sample_ids=None):
    """Persist the pooled topics (+ metadata, + optional JSD matrix) to ``out_dir``.

    The topic x feature matrix goes to **``all_topics_feature.npy``** (float32)
    with its column names in ``feature_ids.csv``, not to CSV. On the HRD-sized
    ensemble (2700 topics x 6866 ASVs) that is the difference between 42.1s /
    254MB and 0.21s / 74MB to write, and 5.0s vs 0.02s to read back. The small
    frames stay CSV so the cache is still greppable by eye.

    ``load_ensemble`` reads either layout, so directories written before this
    change keep working untouched.
    """
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "all_topics_feature.npy"),
            topic_feature_df.values.astype(np.float32))
    pd.DataFrame({"feature_id": list(topic_feature_df.columns)}).to_csv(
        os.path.join(out_dir, "feature_ids.csv"), index=False)
    topic_meta_df.to_csv(os.path.join(out_dir, "all_topics_metadata.csv"), index=False)
    if jsd_dist is not None:
        np.save(os.path.join(out_dir, "jsd_distance_matrix.npy"), jsd_dist)
    if doc_topic is not None:
        np.save(os.path.join(out_dir, "all_topics_doc.npy"),
                np.asarray(doc_topic, dtype=np.float32))
    if sample_ids is not None:
        pd.DataFrame({"sample_id": list(sample_ids)}).to_csv(
            os.path.join(out_dir, "sample_ids.csv"), index=False)
    return out_dir


def load_ensemble(out_dir):
    """Load a cached ensemble. Returns (topic_feature_df, topic_meta_df, jsd_dist).

    Reads the ``.npy`` layout written by :func:`save_ensemble`, falling back to
    the older ``all_topics_feature.csv`` so existing result directories (the HRD
    LDA ensemble among them) load unchanged.

    ``jsd_dist`` is ``None`` if it was never saved (recompute with
    ``archive.compute_jsd_matrix``) — the Hellinger path in ``clustering``
    does not need it.
    """
    topic_feature_df = _load_topic_feature(out_dir)
    topic_meta_df = pd.read_csv(os.path.join(out_dir, "all_topics_metadata.csv"))
    jsd_path = os.path.join(out_dir, "jsd_distance_matrix.npy")
    jsd_dist = np.load(jsd_path) if os.path.exists(jsd_path) else None
    return topic_feature_df, topic_meta_df, jsd_dist


def _load_topic_feature(out_dir):
    """Pooled topic x feature frame from either the ``.npy`` or the legacy CSV layout."""
    npy = os.path.join(out_dir, "all_topics_feature.npy")
    if os.path.exists(npy):
        cols = pd.read_csv(os.path.join(out_dir, "feature_ids.csv"))["feature_id"]
        return pd.DataFrame(np.load(npy), columns=list(cols))
    return pd.read_csv(os.path.join(out_dir, "all_topics_feature.csv"))


# --------------------------------------------------------------------------- #
# Coherence + medoid support (feature-space, used by the current clustering)
# --------------------------------------------------------------------------- #
def topic_coherence_umass(profile_df, V, feature_names, n_top=10):
    """Per-meta-LC ``u_mass`` coherence of the top ``n_top`` features.

    ``u_mass`` scores a topic by how often its top features **co-occur in the
    same sample**, using document frequencies straight off the count table. It is
    chosen over ``c_v`` (which ``evaluate._coherence`` uses) for two reasons:

    * **Cost.** ``c_v`` slides a window over token sequences, and building those
      sequences means expanding counts into repeated tokens — 6.5M tokens for
      HRD, then a sliding window over all of them. ``u_mass`` needs only a
      bag-of-words corpus and one pass of document co-occurrence.
    * **Meaning.** Repeating an ASV *n* times to fake a token stream inflates
      its within-window co-occurrence with itself and its neighbours. Presence
      per sample is the honest unit for abundance data.

    Scores are negative and unbounded below; **less negative is more coherent**.
    They compare topics *within one corpus* and are not portable across datasets.

    Coherence is always computed in **feature space**, against the matrix the
    ensemble was fitted on — regardless of the taxonomic rank a profile is being
    displayed at, since that is a display choice and the topic is a distribution
    over features.

    Parameters
    ----------
    profile_df : pd.DataFrame (n_meta_lc, n_features)
        Feature-level meta-LC profiles; the index labels the returned dict.
    V : array-like (n_samples, n_features)
        The count table, same column order as ``feature_names``.
    feature_names : sequence of str
    n_top : int
        Top features per meta-LC to score.

    Returns
    -------
    dict[str, float] — profile row label -> coherence, rounded to 4 dp.
    """
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel

    feat = [str(f) for f in feature_names]
    V = np.asarray(V)
    dictionary = Dictionary([feat])
    corpus = [
        [(dictionary.token2id[feat[j]], int(V[i, j]))
         for j in np.flatnonzero(V[i] > 0)]
        for i in range(V.shape[0])
    ]
    cols = [str(c) for c in profile_df.columns]
    topics = [
        [cols[j] for j in np.asarray(row).argsort()[-n_top:][::-1]]
        for row in profile_df.values
    ]
    cm = CoherenceModel(topics=topics, corpus=corpus, dictionary=dictionary,
                        coherence="u_mass")
    return {str(lbl): round(float(v), 4)
            for lbl, v in zip(profile_df.index, cm.get_coherence_per_topic())}


def compute_medoids_from_features(
    topic_feature_df,
    labels,
    probabilities=None,
    metric="jensenshannon",
):
    """Cluster medoids without a full distance matrix.

    Distances are computed **only within each cluster** (members-only ``pdist``),
    which is cheap because clusters are small::

        medoid(c) = argmin_i  sum_j  p_j * d(x_i, x_j),   i, j in cluster c

    Called directly by ``clustering.cluster_medoids`` on the **current**
    (Hellinger) path — not legacy, despite living next to the legacy JSD
    functions that used to be here (see ``archive.py``: ``compute_medoids_from_distance``
    is the JSD-matrix equivalent of this, and *is* legacy).

    Returns
    -------
    medoid_indices : list[int]  global row indices, ordered by cluster id.
    """
    from scipy.spatial.distance import pdist, squareform

    X = np.asarray(topic_feature_df.values, dtype=float)
    labels = np.asarray(labels)
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=float)

    cluster_ids = sorted(int(c) for c in set(labels.tolist()) if c >= 0)
    medoid_indices = []
    for cid in cluster_ids:
        member_pos = np.where(labels == cid)[0]
        sub = X[member_pos]
        if sub.shape[0] == 1:
            medoid_indices.append(int(member_pos[0]))
            continue
        D = squareform(pdist(sub, metric=metric))
        if probabilities is not None:
            weights = probabilities[member_pos]
        else:
            weights = np.ones(member_pos.shape[0])
        medoid_local = int(np.argmin(D @ weights))
        medoid_indices.append(int(member_pos[medoid_local]))
    return medoid_indices


# --------------------------------------------------------------------------- #
# Meta-LC composition across k ("stripe-Sankey") — data prep; drawing is
# plotting.plot_meta_stripesankey
# --------------------------------------------------------------------------- #
NOISE_KEY = "noise"


def meta_lc_k_counts(labels, topic_meta_df, include_noise=True, value="topics"):
    """How the pooled topics fitted at each ``k`` distribute over the meta-LCs.

    ``k`` is *not* the number of meta-LCs. The ``n_runs`` fits at a fixed ``k``
    are independent random initialisations, so the ``n_runs * k`` topics they
    produce can land in more (or fewer) than ``k`` clusters — e.g. the 100 topics
    fitted at ``k=2`` may split across 3 meta-LCs. This table is that fact in
    numbers, and the input to :func:`plotting.plot_meta_stripesankey`.

    Parameters
    ----------
    labels : array-like of int, length n_topics
        PLSCAN cluster label per pooled topic (``-1`` = noise), as returned by
        ``Ensemble.labels``.
    topic_meta_df : pd.DataFrame
        The ``k / run / topic / weight`` table from :func:`nmf.run_nmf_ensemble`
        or :func:`lda.run_lda_ensemble`, aligned **row-for-row** with the pooled
        topics (and therefore with ``labels``).
    include_noise : bool
        Keep PLSCAN's ``-1`` topics as a ``"noise"`` column. True by default:
        how much of each k the clustering refuses to assign is part of the story,
        and it is what makes the row totals check out exactly.
    value : {"topics", "runs"}
        ``"topics"`` counts pooled topics (the node size). ``"runs"`` counts the
        *distinct runs* contributing at least one topic — used for hover text, to
        distinguish "one run went haywire and produced 8 topics here" from "8
        different runs agree this meta-LC exists at this k".

    Returns
    -------
    pd.DataFrame
        Rows = ``k`` (ascending), columns = meta-LC id (ascending), plus a
        trailing ``"noise"`` column when ``include_noise``. Missing combinations
        are 0.
    """
    if value not in ("topics", "runs"):
        raise ValueError(f"value must be 'topics' or 'runs', got {value!r}")

    labels = np.asarray(labels)
    if len(labels) != len(topic_meta_df):
        raise ValueError(
            f"labels ({len(labels)}) and topic_meta_df ({len(topic_meta_df)}) "
            "disagree on the number of pooled topics — they must be aligned "
            "row-for-row."
        )

    df = pd.DataFrame({
        "k": np.asarray(topic_meta_df["k"]),
        "run": np.asarray(topic_meta_df["run"]),
        "lc": labels,
    })
    if not include_noise:
        df = df[df["lc"] >= 0]

    if value == "topics":
        tab = pd.crosstab(df["k"], df["lc"])
    else:
        tab = (df.groupby(["k", "lc"])["run"].nunique()
                 .unstack(fill_value=0))

    # -1 -> a named trailing column, so meta-LC ids stay integers and sortable
    cols = sorted(c for c in tab.columns if c != -1)
    tab = tab.reindex(columns=cols + ([-1] if -1 in tab.columns else []), fill_value=0)
    tab = tab.rename(columns={-1: NOISE_KEY})
    tab.index.name = "k"
    tab.columns.name = "meta_lc"
    return tab.sort_index()
