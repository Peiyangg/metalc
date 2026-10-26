"""All drawing code for the ensemble pipeline — method-agnostic, called by
``ensemble.py`` (``ProfilePlot`` / ``stripesankey``) and by ``nmf.py``'s old
fixed-W heatmap path. Nothing here knows whether the topics it is drawing came
from LDA or NMF.

Two functions (``plot_top_taxa``, ``plot_projection``) need label/order-building
helpers that live on ``ensemble.py`` (``lc_cluster_order`` / ``build_cluster_labels``)
— imported lazily inside the function, the same way ``ensemble.py`` imports back
from here, so the two files can both be imported in either order.
"""

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Cluster-id parsing + the two live helpers plot_cluster_profile needs.
# Moved out of the otherwise-dead ensemble_topic.py because plot_top_taxa
# (below) and therefore every ``metalc.plot_meta(view="profile")`` call still
# goes through them.
# --------------------------------------------------------------------------- #
def _parse_cluster_id(cl):
    """Parse an integer cluster id out of a row label.

    Emits are all ``'MetaLC-3'`` now, but the older spellings are still parsed:
    result CSVs written before the rename, and anything downstream that stored a
    label rather than an id. Accepts int, ``'MetaLC-3'``, ``'Meta MC-3'``,
    ``'Meta MC-3'``, ``'MC-3'``, ``'Meta LC-3'``, ``'LC-3'``, or legacy
    ``'cluster_3'``.
    """
    if isinstance(cl, str):
        # Normalise non-breaking space to regular space for parsing
        s = cl.replace(' ', ' ')
        if s.startswith('MetaLC-'):
            return int(s.split('-', 1)[1])
        if s.startswith('Meta MC-') or s.startswith('Meta LC-'):
            return int(s.split('-', 1)[1])
        if s.startswith('MC-') or s.startswith('LC-'):
            return int(s.split('-', 1)[1])
        if s.startswith('cluster_'):
            return int(s.split('_', 1)[1])
    return int(cl)


def compute_cluster_stability(labels, topic_meta_df, n_runs=50):
    """Compute first-k and stable-k for each cluster.

    first_k: earliest K where this cluster contains any topic.
    stable_k: first K where >n_runs/2 distinct runs contribute a topic.

    Returns (first_k_map, stable_k_map) — dicts keyed by cluster id.
    """
    valid_mask = labels >= 0
    valid_labels = labels[valid_mask]
    valid_meta = topic_meta_df.iloc[np.where(valid_mask)[0]]

    first_k_map = {}
    stable_k_map = {}

    for cid in sorted(set(valid_labels)):
        sub = valid_meta[valid_labels == cid]
        ks_in = sorted(sub['k'].unique())
        first_k_map[int(cid)] = int(ks_in[0])
        stable = None
        for kv in ks_in:
            n_runs_with_topic = sub.loc[sub['k'] == kv, 'run'].nunique()
            if n_runs_with_topic > n_runs / 2:
                stable = int(kv)
                break
        stable_k_map[int(cid)] = stable

    return first_k_map, stable_k_map


def k_dist_cluster_order(labels, topic_meta_df, first_k_map, stable_k_map):
    """Order clusters by (first_k, peak_k, stable_k) tuple.

    Returns list of "MetaLC-<id>" strings in sorted order.
    """
    valid_mask = labels >= 0
    valid_labels = labels[valid_mask]
    valid_meta = topic_meta_df.iloc[np.where(valid_mask)[0]]
    cluster_ids = sorted(set(valid_labels))

    peak_k_map = {}
    for cid in cluster_ids:
        sub = valid_meta[valid_labels == cid]
        counts = sub['k'].value_counts()
        peak_k_map[int(cid)] = int(counts.idxmax())

    sort_data = []
    for cid in cluster_ids:
        cid_int = int(cid)
        first = first_k_map.get(cid_int, 999)
        peak = peak_k_map.get(cid_int, 999)
        stable = stable_k_map.get(cid_int, 999) or 999
        sort_data.append((first, peak, stable, cid_int))

    sort_data.sort()
    return [f"MetaLC-{item[3]}" for item in sort_data]


def _row_label(cluster_label, row_annotations):
    """Row label for a profile panel, with an optional annotation on a second line.

    Used to hang a coherence score off each meta-LC without changing how the
    label is parsed elsewhere — `_parse_cluster_id` reads the first segment.
    """
    if not row_annotations or cluster_label not in row_annotations:
        return cluster_label
    return f"{cluster_label}\n{row_annotations[cluster_label]}"


def plot_cluster_profile(
    df,
    title="Cluster Profile",
    figsize=None,
    col_font_size=10,
    tick_font_size=8,
    label_font_size=9,
    title_font_size=12,
    row_height=0.6,
    col_width=1.2,
    prefix_strip=None,
    bar_height=0.6,
    save_path=None,
    dpi=300,
    cluster_order=None,
    labels=None,
    topic_meta_df=None,
    n_runs=50,
    show_bar_values=False,
    row_annotations=None,
):
    """Small multiples grid of horizontal bar charts for medoid profiles.

    When labels and topic_meta_df are provided, a K-distribution line chart
    is shown after the cluster label column. First-k (red dot) and
    stable-k (blue square) are computed internally.

    Parameters
    ----------
    df : pd.DataFrame
        Rows = clusters, columns = features.
    cluster_order : list or None
        Custom cluster order (row names). If None, uses JSD hierarchical clustering.
    labels, topic_meta_df : optional
        Pass both to embed K-distribution line charts.
    n_runs : int
        Number of runs per K (used to compute stable-k threshold = n_runs/2).
    show_bar_values : bool
        If True, show the numeric value to the left of each bar.
    """
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, leaves_list

    has_kdist = labels is not None and topic_meta_df is not None

    df_t = df.T

    if cluster_order is None:
        data_norm = df.div(df.sum(axis=1), axis=0).fillna(0)
        if len(data_norm) > 2:
            def _jsd(p, q):
                eps = 1e-10
                p = np.array(p) + eps
                q = np.array(q) + eps
                p = p / p.sum()
                q = q / q.sum()
                m_val = 0.5 * (p + q)
                return 0.5 * np.sum(p * np.log(p / m_val)) + 0.5 * np.sum(q * np.log(q / m_val))
            dist = pdist(data_norm.values, metric=_jsd)
            Z = linkage(dist, method='average')
            cluster_order = [data_norm.index[i] for i in leaves_list(Z)]
        else:
            cluster_order = list(data_norm.index)

    feature_order = df_t.max(axis=1).sort_values(ascending=False).index.tolist()

    n_clusters = len(cluster_order)
    n_features = len(feature_order)

    # Column layout: [kdist?] [feature cols...]
    # Cluster label is rendered as ylabel on the leftmost axes
    n_cols = n_features + (1 if has_kdist else 0)
    feat_col_offset = 1 if has_kdist else 0

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
    ]

    dominant = {}
    for cl in cluster_order:
        top_feat = df.loc[cl].idxmax()
        dominant[cl] = top_feat.replace(prefix_strip, '', 1) if prefix_strip else top_feat

    if has_kdist:
        ks = sorted(topic_meta_df['k'].unique())
        kdist_counts = {}
        first_k_map = {}
        stable_k_map = {}
        for cl in cluster_order:
            cl_id = _parse_cluster_id(cl)
            sub_mask = labels == cl_id
            sub_meta = topic_meta_df.iloc[np.where(sub_mask)[0]]
            counts = sub_meta['k'].value_counts().reindex(ks, fill_value=0).sort_index()
            kdist_counts[cl] = counts

            ks_present = sorted(sub_meta['k'].unique())
            first_k_map[cl_id] = int(ks_present[0]) if len(ks_present) > 0 else None
            stable = None
            for kv in ks_present:
                if sub_meta.loc[sub_meta['k'] == kv, 'run'].nunique() > n_runs / 2:
                    stable = int(kv)
                    break
            stable_k_map[cl_id] = stable

        kdist_y_max = max(c.max() for c in kdist_counts.values()) * 1.1
        # Pick a single round y-tick near the midpoint
        raw_mid = kdist_y_max / 2
        for step in [50, 25, 20, 10, 5, 2, 1]:
            candidate = int(round(raw_mid / step)) * step
            if candidate > 0:
                kdist_y_tick = candidate
                break
        else:
            kdist_y_tick = max(1, int(round(raw_mid)))

    if figsize is None:
        figsize = (col_width * n_cols + 2, row_height * n_clusters + 1.5)

    fig, axes = plt.subplots(
        n_clusters, n_cols, figsize=figsize, squeeze=False,
    )

    x_max = df.values.max() * 1.1

    for r_idx, cl in enumerate(cluster_order):
        if has_kdist:
            # Column 0: K-distribution line chart
            ax_k = axes[r_idx, 0]
            counts = kdist_counts[cl]
            ax_k.plot(ks, counts.values, color='#555555', linewidth=1.2, marker='o', markersize=3, zorder=2)
            ax_k.fill_between(ks, counts.values, alpha=0.15, color='#555555')
            ax_k.set_ylim(0, kdist_y_max)
            ax_k.set_xlim(ks[0] - 0.3, ks[-1] + 0.3)
            ax_k.set_yticks([kdist_y_tick])
            ax_k.tick_params(axis='y', labelsize=tick_font_size - 1)
            ax_k.tick_params(axis='x', labelsize=tick_font_size - 1)
            ax_k.set_ylabel(_row_label(cl, row_annotations), fontsize=tick_font_size,
                            rotation=0, labelpad=60, va='center')

            cl_id = _parse_cluster_id(cl)
            fk = first_k_map.get(cl_id)
            if fk is not None and fk in counts.index:
                ax_k.plot(fk, counts[fk], 'o', color='red', markersize=5, zorder=3)

            if r_idx < n_clusters - 1:
                ax_k.tick_params(axis='x', labelbottom=False)
            if r_idx == 0:
                ax_k.set_title("K dist", fontsize=col_font_size)

        # Feature columns
        for c_idx, feat in enumerate(feature_order):
            ax = axes[r_idx, c_idx + feat_col_offset]
            val = df.loc[cl, feat]
            ax.barh(0, val, color=colors[c_idx % len(colors)], height=bar_height)
            ax.set_xlim(0, x_max)
            ax.set_yticks([])
            ax.tick_params(axis='x', labelsize=tick_font_size)

            if show_bar_values and val > 0:
                ax.text(val + x_max * 0.02, 0, f"{val:.2f}", va='center', ha='left',
                        fontsize=tick_font_size - 1, color='#333333')

            if r_idx < n_clusters - 1:
                ax.tick_params(axis='x', labelbottom=False)
            if r_idx == 0:
                feat_label = feat.replace(prefix_strip, '', 1) if prefix_strip else feat
                ax.set_title(feat_label, fontsize=col_font_size, rotation=0)
            if not has_kdist and c_idx == 0:
                ax.set_ylabel(_row_label(cl, row_annotations), fontsize=tick_font_size,
                              rotation=0, labelpad=60, va='center')

        axes[r_idx, -1].annotate(
            dominant[cl],
            xy=(1.05, 0.5), xycoords='axes fraction',
            fontsize=label_font_size, va='center', ha='left',
        )

    fig.suptitle(title, fontsize=title_font_size, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig


def plot_datamapplot(embedding, labels, point_labels, highlight_label_list,
                     title="Ensemble LDA Topics",
                     sub_title="UMAP + HDBSCAN clustering",
                     font_size=12,
                     font_weight=400,
                     point_size=None,
                     figsize=(14, 14),
                     cluster_color_map=None):
    """Datamapplot scatter of the UMAP embedding.

    Parameters
    ----------
    cluster_color_map : dict {cluster_id: hex_color}, optional
        If provided, points are coloured by this map (cluster_id -> hex string)
        instead of the default tab20 palette. Noise (-1) stays grey. Labels and
        layout are unchanged.
    """
    import datamapplot
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if cluster_color_map is None:
        cmap = plt.cm.tab20
        color_array = np.array([
            "#999999" if cl == -1
            else mcolors.to_hex(cmap(int(cl) % 20))
            for cl in labels
        ])
    else:
        color_array = np.array([
            "#999999" if cl == -1
            else cluster_color_map.get(int(cl), "#999999")
            for cl in labels
        ])

    fig, ax = datamapplot.create_plot(
        embedding,
        point_labels,
        title=title,
        sub_title=sub_title,
        noise_label="Unlabelled",
        noise_color="#999999",
        highlight_labels=highlight_label_list,
        marker_color_array=color_array,
        use_medoids=True,
        label_over_points=True,
        dynamic_label_size=True,
        max_font_size=font_size + 1,
        min_font_size=font_size,
        max_font_weight=font_weight + 1,
        min_font_weight=font_weight,
        label_wrap_width=22,
        point_size=point_size,
        figsize=figsize,
        force_matplotlib=True,
    )
    return fig, ax


# --------------------------------------------------------------------------- #
# Moved from ensemble.py — fixed-factor loadings, meta-LC profiles, stripesankey.
# --------------------------------------------------------------------------- #
def plot_loadings_heatmap(
    loadings,
    cluster_labels=None,
    sample_ids=None,
    normalize="row",
    sort_by_dominant=True,
    title="Sample x meta-LC loadings (fixed-W GKL inference)",
    cmap="magma",
    figsize=None,
    save_path=None,
    dpi=200,
):
    """Heatmap of the fixed-topic loadings (samples x meta-LC).

    Parameters
    ----------
    loadings : np.ndarray (n_samples, n_topics)
        Output of :func:`nmf.infer_loadings_fixed_topics`.
    cluster_labels : sequence of str or None
        Column labels (e.g. the meta-LC profile index). Defaults to LC-0..LC-{n}.
    sample_ids : sequence or None
        Row labels; only drawn when there are few samples.
    normalize : {"row", None}
        "row" -> each sample's loadings scaled to sum 1 (mixing proportions).
    sort_by_dominant : bool
        Order samples by their dominant meta-LC, then by that weight (like
        ``nmf.normalize(cluster=True)``), so block structure is visible.
    """
    import matplotlib.pyplot as plt

    M = np.asarray(loadings, dtype=float)
    if normalize == "row":
        rs = M.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        M = M / rs

    n_samples, n_topics = M.shape
    if sort_by_dominant:
        dominant = np.argmax(M, axis=1)
        weight = np.max(M, axis=1)
        order = np.lexsort((-weight, dominant))
        M = M[order]
        if sample_ids is not None:
            sample_ids = np.asarray(sample_ids)[order]

    if cluster_labels is None:
        cluster_labels = [f"LC-{i}" for i in range(n_topics)]

    if figsize is None:
        figsize = (max(4, n_topics * 0.7 + 2), max(4, n_samples * 0.05 + 2))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(n_topics))
    ax.set_xticklabels(cluster_labels, rotation=45, ha="right", fontsize=8)
    if sample_ids is not None and n_samples <= 60:
        ax.set_yticks(range(n_samples))
        ax.set_yticklabels(sample_ids, fontsize=6)
    else:
        ax.set_yticks([])
        ax.set_ylabel(f"{n_samples} samples")
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                 label="loading" + (" (row-normalised)" if normalize == "row" else ""))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_top_taxa(
    profile_df,
    labels,
    topic_meta_df,
    order_by="first_k",
    feature_cutoff=0.05,
    n_runs=50,
    title="meta-LC profiles",
    save_path=None,
    row_annotations=None,
):
    """Small-multiples bar chart of each meta-LC's top features.

    ``order_by``:
      - ``"first_k"``  order rows by K-emergence (first_k / stable_k).
      - ``"clustering"`` order rows by a JSD hierarchical clustering of the
        meta-LC profiles (``plot_cluster_profile`` does this when cluster_order
        is None).
    """
    import ensemble as E

    if order_by == "first_k":
        first_k, stable_k = compute_cluster_stability(labels, topic_meta_df, n_runs=n_runs)
        cluster_order = E.lc_cluster_order(labels, topic_meta_df, first_k, stable_k)
    elif order_by == "clustering":
        cluster_order = None
    else:
        raise ValueError(f"order_by must be 'first_k' or 'clustering', got {order_by!r}")

    top = profile_df.columns[profile_df.max() > feature_cutoff].tolist()
    return plot_cluster_profile(
        profile_df[top], title=title, col_font_size=7,
        cluster_order=cluster_order, labels=labels, topic_meta_df=topic_meta_df,
        show_bar_values=True, save_path=save_path,
        row_annotations=row_annotations,
    )


def plot_projection(
    topic_feature_df,
    labels,
    profile_df,
    n_clusters,
    umap_random_state=42,
    feature_cutoff=0.05,
    title="Ensemble topics — PLSCAN clusters",
    sub_title="UMAP of the Hellinger embedding (same geometry as the clustering)",
    save_path=None,
):
    """UMAP overview of the pooled topics, coloured by cluster and labelled by
    each meta-LC's top features.

    The embedding comes from :func:`clustering.project_topics`, which runs
    the *same* two steps the clustering does — ``sqrt(p)``, then euclidean — and
    only swaps PLSCAN for a 2-D UMAP. So this is a projection of the very space
    the clusters were formed in. The geometry is deliberately not a parameter:
    a projection in a different distance can put points inside the wrong
    cluster's colour, which would misrepresent the clustering the figure exists
    to show. For an overview in some other geometry, call
    :func:`archive.run_umap_features` directly.
    """
    import ensemble as E
    from clustering import project_topics

    embedding = project_topics(topic_feature_df, random_state=umap_random_state)
    point_labels, highlight = E.build_cluster_labels(
        profile_df, n_clusters, labels, feature_cutoff=feature_cutoff,
    )
    fig, _ = plot_datamapplot(
        embedding, labels, point_labels, highlight,
        title=title, sub_title=sub_title,
        font_size=10, font_weight=600, point_size=20,
    )
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def _scurve(x0, y0, x1, y1, n=32):
    """Sankey-style S-curve between two points (flat at both ends)."""
    t = np.linspace(0.0, 1.0, n)
    return x0 + (x1 - x0) * t, y0 + (y1 - y0) * (3 * t ** 2 - 2 * t ** 3)


def plot_meta_stripesankey(
    counts,
    runs=None,
    column_totals=None,
    title="Meta-LC composition across k",
    link_width=2,
    normalize=False,
    node_labels=None,
    node_width=0.24,
    gap_frac=0.015,
    palette=None,
    width=None,
    height=760,
    save_path=None,
):
    """Stripe + Sankey of how the ensemble's topics distribute over meta-LCs as
    ``k`` grows. Returns a plotly ``Figure``.

    One column per ``k``; within a column one rectangle per meta-LC, **its height
    the raw number of topics** fitted at that ``k`` that belong to it. Column
    totals therefore grow as ``n_runs * k`` — the funnel is the data, not a bug
    (pass ``normalize=True`` to level the columns and compare shares instead).

    Each meta-LC keeps one colour and one vertical slot in every column, and thin
    constant-width connectors link a meta-LC to itself in the next column, so
    appearing / persisting / vanishing reads straight off the picture.

    The connectors deliberately carry **no magnitude** — all quantity is in the
    node heights. They are drawn by :func:`_scurve` from ``_links`` below; swap
    that one loop to give the ribbons a real flow semantics later.

    Notes
    -----
    This is not a ``plotly.graph_objects.Sankey``. That trace derives node height
    from ``max(sum incoming, sum outgoing)`` link value, so node size cannot be
    set independently of ribbon width — exactly what this figure needs. The nodes
    and connectors are therefore drawn directly.

    Parameters
    ----------
    counts : pd.DataFrame
        From :func:`ensemble.meta_lc_k_counts` — rows ``k``, columns meta-LC
        (+ ``"noise"``).
    runs : pd.DataFrame, optional
        Same shape, from ``ensemble.meta_lc_k_counts(..., value="runs")``. Adds a
        "from N of the runs" line to each node's hover.
    column_totals : pd.Series, optional
        True number of topics fitted at each ``k`` (``n_runs * k``), indexed like
        ``counts``. Needed when ``counts`` has had its ``noise`` column dropped:
        without it the shares and the per-column caption would silently be taken
        over the *assigned* topics only, so a column would claim "95 topics" when
        200 were fitted and 105 went unassigned. Defaults to the row sums of
        ``counts`` (correct when nothing has been dropped).
    title : str
    link_width : float
        Connector width in pixels (constant).
    normalize : bool
        Divide each column by its total, so every k column is the same height and
        the rectangles read as shares. Hover always reports the raw counts.
    node_labels : dict, optional
        ``{meta_lc_id: str}`` short label drawn on the node (e.g. its top genus).
        Only drawn where the rectangle is tall enough to fit the text.
    node_width : float
        Rectangle width, in units of the column spacing (columns sit at x = 0, 1, …).
    gap_frac : float
        Vertical gap between stacked rectangles, as a fraction of the tallest
        column's total.
    palette : sequence of str, optional
        Colours cycled over meta-LC ids. Defaults to plotly's Dark24.
    width, height : int
        Figure size in px. ``width`` defaults to a size that scales with the
        number of columns.
    save_path : str, optional
        ``.html`` -> ``write_html`` (interactive, self-contained).
        ``.png`` / ``.pdf`` / ``.svg`` -> ``write_image`` (needs kaleido).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    from ensemble import NOISE_KEY

    if counts.empty:
        raise ValueError("counts is empty — nothing to draw.")

    ks = list(counts.index)
    lcs = list(counts.columns)
    if len(ks) < 2:
        raise ValueError(
            f"the stripe-Sankey needs at least 2 k columns to link, got {ks}."
        )

    if palette is None:
        from plotly.colors import qualitative
        palette = qualitative.Dark24
    color = {}
    i = 0
    for lc in lcs:
        if lc == NOISE_KEY:
            color[lc] = "#b0b0b0"
        else:
            color[lc] = palette[i % len(palette)]
            i += 1

    # ---- geometry ---------------------------------------------------------
    drawn = counts.sum(axis=1)                 # what the rectangles add up to
    totals = drawn if column_totals is None else pd.Series(column_totals).reindex(drawn.index)
    if totals.isna().any():
        raise ValueError("column_totals must cover every k in counts.index")
    # normalise by the drawn total so the columns level exactly
    heights = counts.div(drawn.replace(0, 1), axis=0) if normalize else counts
    gap = float(heights.sum(axis=1).max()) * gap_frac
    half = node_width / 2.0

    # rect[(k, lc)] = (y_bottom, y_top); stacked so meta-LC ids run top-to-bottom
    rect = {}
    for xi, k in enumerate(ks):
        present = [lc for lc in lcs if counts.at[k, lc] > 0]
        span = float(heights.loc[k, present].sum()) + gap * max(len(present) - 1, 0)
        y = -span / 2.0
        for lc in reversed(present):          # build upward -> first id on top
            h = float(heights.at[k, lc])
            rect[(k, lc)] = (y, y + h)
            y += h + gap

    # ---- nodes ------------------------------------------------------------
    fig = go.Figure()
    seen = set()
    for xi, k in enumerate(ks):
        for lc in lcs:
            if (k, lc) not in rect:
                continue
            y0, y1 = rect[(k, lc)]
            n_top = int(counts.at[k, lc])
            share = n_top / totals[k] * 100 if totals[k] else 0.0
            hover = (f"<b>{'noise' if lc == NOISE_KEY else f'MetaLC-{lc}'}</b>"
                     f"<br>k = {k}"
                     f"<br>{n_top} topics ({share:.1f}% of k={k})")
            if runs is not None and lc in runs.columns and k in runs.index:
                hover += f"<br>from {int(runs.at[k, lc])} distinct runs"
            fig.add_trace(go.Scatter(
                x=[xi - half, xi + half, xi + half, xi - half, xi - half],
                y=[y0, y0, y1, y1, y0],
                fill="toself", fillcolor=color[lc],
                line=dict(width=0), mode="lines",
                hoveron="fills", hoverinfo="text", text=hover,
                legendgroup=str(lc), name=("noise" if lc == NOISE_KEY else f"MetaLC-{lc}"),
                showlegend=lc not in seen,
            ))
            seen.add(lc)
            if node_labels and lc in node_labels and (y1 - y0) > gap * 3:
                fig.add_annotation(
                    x=xi, y=(y0 + y1) / 2, text=str(node_labels[lc]),
                    showarrow=False, font=dict(size=9, color="white"),
                )

    # ---- connectors: same meta-LC, adjacent columns, constant width -------
    for lc in lcs:
        xs, ys = [], []
        for xi in range(len(ks) - 1):
            a, b = (ks[xi], lc), (ks[xi + 1], lc)
            if a not in rect or b not in rect:
                continue
            ya = sum(rect[a]) / 2.0
            yb = sum(rect[b]) / 2.0
            cx, cy = _scurve(xi + half, ya, xi + 1 - half, yb)
            xs.extend(cx.tolist() + [None])
            ys.extend(cy.tolist() + [None])
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=color[lc], width=link_width),
                opacity=0.55, hoverinfo="skip",
                legendgroup=str(lc), showlegend=False,
            ))

    # ---- axes / chrome ----------------------------------------------------
    unit = "share of k" if normalize else "topics"
    for xi, k in enumerate(ks):
        # when noise has been dropped, say so rather than passing the assigned
        # count off as the number of topics fitted at this k
        cap = (f"{int(totals[k])} topics" if drawn[k] == totals[k]
               else f"{int(drawn[k])} of {int(totals[k])}")
        fig.add_annotation(
            x=xi, y=0, yref="paper", yanchor="top", yshift=-8,
            text=cap, showarrow=False,
            font=dict(size=10, color="#666"),
        )
    fig.update_xaxes(
        tickmode="array", tickvals=list(range(len(ks))),
        ticktext=[f"k = {k}" for k in ks], side="top",
        showgrid=False, zeroline=False,
        range=[-0.5, len(ks) - 0.5],
    )
    fig.update_yaxes(visible=False, showgrid=False, zeroline=False)
    fig.update_layout(
        title=dict(text=f"{title}<br><sub>node height = {unit}</sub>", x=0.5),
        width=width if width is not None else max(900, 150 * len(ks) + 280),
        height=height,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=40, r=40, t=110, b=60),
        legend=dict(title="meta-LC", itemsizing="constant"),
        hovermode="closest",
    )

    if save_path:
        if save_path.lower().endswith(".html"):
            fig.write_html(save_path, include_plotlyjs="cdn")
        else:
            fig.write_image(save_path, scale=2)
    return fig


def split_metadata_columns(md, exclude=(), max_categories=15):
    """Auto-split metadata columns into (categorical, continuous) for the heatmap.

    Numeric dtype -> continuous (gradient bar). Non-numeric with at most
    ``max_categories`` distinct values -> categorical (annotation bar). Columns
    with too many categories are skipped (they'd make an unreadable rainbow).
    """
    universal, continuous = [], []
    for col in md.columns:
        if col in exclude:
            continue
        s = md[col]
        if pd.api.types.is_numeric_dtype(s):
            continuous.append(col)
        elif s.nunique(dropna=True) <= max_categories:
            universal.append(col)
    return universal, continuous


def _prepare_metadata(metadata, sample_ids, universal_headers, continuous_headers,
                      id_column="ID"):
    """Align a metadata DataFrame to the samples and pick annotation columns.

    Returns (md, universal_headers, continuous_headers) ready for the viz: index
    is the sample id (neutral name), ``id_column`` is a real column, categoricals
    are string ('Missing'-filled), continuous are numeric. Headers are
    auto-detected via ``split_metadata_columns`` when not given.
    """
    md = metadata.copy()
    # Make the sample id the index: use it if the index already matches, else
    # find the column that overlaps the sample ids.
    if len(set(sample_ids) & set(md.index.astype(str))) < len(sample_ids) * 0.5:
        for col in md.columns:
            if len(set(sample_ids) & set(md[col].astype(str))) >= len(sample_ids) * 0.5:
                md = md.set_index(col)
                break
    md.index = md.index.astype(str)
    md = md.reindex(md.index.union([str(s) for s in sample_ids]))

    if universal_headers is None and continuous_headers is None:
        universal_headers, continuous_headers = split_metadata_columns(
            md, exclude=(id_column,),
        )
    universal_headers = universal_headers or []
    continuous_headers = continuous_headers or []

    md[id_column] = md.index
    for h in universal_headers:
        if h in md.columns:
            md[h] = md[h].fillna("Missing").astype(str)
    for h in continuous_headers:
        if h in md.columns:
            md[h] = pd.to_numeric(md[h], errors="coerce")

    headers = [h for h in universal_headers if h in md.columns] or [id_column]
    cont = [h for h in continuous_headers if h in md.columns]

    md.index = list(md[id_column])
    md.index.name = "sample_index"
    return md, headers, cont


def render_fixedw_heatmap(
    H_norm,
    W_df,
    sample_ids,
    metadata,
    out_dir,
    group_map=None,
    universal_headers=None,
    continuous_headers=None,
    id_column="ID",
    order_by="heatmap_clustering",
):
    """Write the lda4microbiome layout and render the annotated H heatmap.

    Encapsulates all the lda4microbiome plumbing: exports H (sample x meta-LC),
    W (meta-LC x feature) and a minimal taxonomy, attaches ``metadata`` (aligned +
    auto-typed), and returns the interactive heatmap (also saved as HTML in
    ``out_dir``).

    Renders via ``lda4microbiome.heatmap_sample``, which takes the sample x MC
    matrix directly. It replaced ``LDAModelVisualizerInteractive``, which read
    the on-disk layout and returned a plotly figure. Two consequences:

    * the return value is an **anywidget**, not a ``go.Figure`` — display it in a
      notebook cell; size it with ``height`` rather than ``update_layout``;
    * the ``lda_results/`` + ``intermediate/`` layout below is no longer needed
      to draw the figure. It is still written, because it is the documented
      interchange format for this repo and other tools read it.

    ``order_by`` keeps its old spelling: ``"heatmap_clustering"`` (or None) sorts
    samples by the sample-MC relationship, anything else is a metadata column.
    """
    import os
    from lda4microbiome import heatmap_sample
    from lda4microbiome.stripesankey import save_widget_html

    topic_labels = list(W_df.index)
    k = len(topic_labels)
    os.makedirs(os.path.join(out_dir, "lda_results", "MC_Sample"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "lda_results", "MC_Feature"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "intermediate"), exist_ok=True)

    pd.DataFrame(H_norm.T, index=topic_labels, columns=list(sample_ids)).to_csv(
        os.path.join(out_dir, "lda_results", "MC_Sample",
                     f"MC_Sample_probabilities{k}.csv")
    )
    W_df.to_csv(
        os.path.join(out_dir, "lda_results", "MC_Feature",
                     f"MC_Feature_Probabilities_{k}.csv")
    )
    feats = list(W_df.columns)
    genera = [group_map.get(f, f) for f in feats] if group_map is not None else feats
    pd.DataFrame({"Genus": genera}, index=feats).to_csv(
        os.path.join(out_dir, "intermediate", "intermediate_taxa.csv")
    )

    md, headers, cont = _prepare_metadata(
        metadata, sample_ids, universal_headers, continuous_headers, id_column,
    )
    meta_path = os.path.join(out_dir, "metadata_viz.csv")
    md.to_csv(meta_path)

    H_df = pd.DataFrame(H_norm, index=[str(s) for s in sample_ids],
                        columns=topic_labels)
    # heatmap_sample infers categorical vs continuous per column itself, so the
    # two header lists collapse into one `include`
    include = [h for h in list(headers) + list(cont) if h in md.columns]
    sort = None if order_by in (None, "heatmap_clustering") else order_by

    fig = heatmap_sample(H_df, md, include=include, sort=sort, height=700)
    save_widget_html(fig, os.path.join(out_dir, f"sample_metaLC_fixedW_k{k}.html"))
    return fig
