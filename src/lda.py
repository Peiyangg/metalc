"""LDA-specific code: the gensim variational-Bayes ensemble fitter (the LDA twin
of ``nmf.run_nmf_ensemble``), the fixed-W LDA solver, and the Guided-vs-Standard
track's plain-LDA primitives.

    topic_feature_df, topic_meta_df = run_lda_ensemble(V, feature_names, [2, 10])

Fits ``n_runs`` models at every ``k`` and pools every topic into one
``(n_topics, n_features)`` frame, with ``(k, run, topic, weight)`` alongside —
byte-for-byte the same contract the NMF track returns, so the shared clustering
in ``clustering`` sees the two tracks identically.

Convergence defaults (measured on HRD, 289 samples x 6866 ASVs, ~22k counts per
sample — i.e. very long "documents")
-------------------------------------------------------------------------------
``iterations`` is the per-document E-step cap; ``gamma_threshold`` is what should
actually stop it. At gensim's default ``iterations=50`` it does not:

    iterations=50   ->  68/289 documents converged by the last pass
    iterations=400  -> 267/289

and the topics that fall out are *materially* different — max matched Hellinger
distance 0.709 between ``iterations=50`` and ``iterations=400`` at k=10 (0.190
for 200 vs 400). That is not numerical noise, it is a different answer, and
pooling under-converged topics hands PLSCAN spread it will read as structure.

Raising the cap is cheap (44s -> 71s per fit, 1.6x) precisely because
``gamma_threshold`` retires each document as soon as it converges: the extra
budget is only spent where it was needed. So ``iterations`` is the knob to
over-provision — it adapts per dataset and per document on its own, unlike
``passes``, which has no stopping rule and must be chosen.

``passes`` is **not** a fixed number any more. It used to be 100, chosen because
max matched Hellinger was only 0.064 between 50 and 100 passes. That argument
turns out to measure the wrong thing: gensim's own ``topic diff`` — the mean
absolute change in ``Elogbeta`` per pass, which it computes and logs but never
exposes — shows the model is still moving at pass 100. Measured on HRD:

    k=2   topic_diff/peak 1.16e-03 at pass 100     not settled
    k=5                   1.61e-03                 not settled
    k=10                  3.81e-04 (crossed 1e-3 at pass 64)   settled

and the topics keep moving with it — max matched Hellinger against the 100-pass
fit is 0.11 at 200 passes, 0.17-0.20 at 400, and 0.44 at 800 for k=2. Letting
each fit stop when *it* settles asks for 161-515 passes on HRD at k=2..3.

So each fit now runs until ``topic diff`` has sat below ``rel_tol`` of its own
peak for ``PATIENCE`` passes, up to a ``max_passes`` ceiling, and every fit
records what it did (see :func:`_report_convergence`). ``passes=100`` is still
accepted and reproduces the old ensembles bit-identically.

``topic diff`` here means the change between the **ends of consecutive passes**,
computed by :func:`_fit_batch` — not the number gensim logs. gensim logs one per
M-step, i.e. one per *chunk*, and a table larger than ``chunksize`` is fitted
online with several M-steps per pass (SO-CPR: 55,886 segments, 28 chunks). There
the logged value is chunk-to-chunk sampling noise scaled by ``rho``, so it
decays like ``rho`` rather than like the model: measured ~pass^-0.38 on SO-CPR,
3.6e-3 to 9.9e-3 of peak at pass 100, which would take 3k-82k passes to reach
1e-3 — while the k=2 topics moved only 0.02-0.03 between pass 100 and 300.
With one chunk per pass the two quantities are the same number, so for tables
like HRD nothing changes, down to the stopping pass.

Even per pass, ``rel_tol`` turned out **not to be a reliable stop** on SO-CPR.
``topic diff`` is a mean over every topic x taxon cell, so one topic drifting
slowly is averaged away. Online fits (k=2/5/10, two seeds each, 600 passes)
crossed 1e-3 of peak at passes 16-56 while their topics were still 0.07-0.27
max matched Hellinger from where they were at pass 600; no single ``rel_tol``
fixed it across k (1e-4 left 0.02-0.22, 2e-5 never triggered at k=2 by 600).
The rule that tracked the topics is the obvious one — **how far has any topic
moved over the last** ``move_window`` **passes** (``move_tol``): at 0.01 over
100 passes it stopped those fits at 320-530 passes, 0.005-0.014 from pass 600.
It is opt-in (``move_tol=None`` keeps ``rel_tol``, so HRD is untouched). No
local rule anticipates a late reorganisation: one k=5 fit sat at 0.011 per 50
passes around pass 200, then moved 0.10 between 350 and 450.

The one caveat the old note got right: ``rho`` decays monotonically, so extra
passes cannot destabilise a fit — which is why ``max_passes`` can be set high.

``eval_every=None`` is not a tuning choice: gensim's default recomputes
perplexity every 10 updates, which costs a full extra E-step pass and changes
nothing. Turning it off measured 12.3s -> 3.2s (3.8x).

``update_every=0`` (batch VB) is deliberately *not* used — it measured ~25%
slower here (3.6s vs 2.8s) and buys only cleaner rho semantics.

Parallelism
-----------
``workers`` fits **independent models in separate processes**. This is not a
gensim feature and does not need to be: the fits share no state, and worker
output is bit-identical to fitting serially (verified: max abs difference 0.0
in both topics and doc-topics for the same seed).

Note this is *not* ``LdaMulticore``, which is the wrong tool twice over here.
It splits work by chunk, and a microbiome table is one chunk (289 documents <
``chunksize``), so it would have nothing to distribute; and it merges worker
results in completion order (``while not result_queue.empty(): other.merge(...)``),
so its result depends on scheduling — which would break the ensemble's premise
that the seed is the only thing that varies.

Jobs are handed out **in batches, one batch per worker**, so the corpus is
pickled once per worker rather than once per fit. That distinction is the whole
speedup: per-fit dispatch measured 1.3x, batched dispatch 3.5x (327s -> 93s for
30 fits).

On macOS numpy links against Accelerate, which ignores ``OMP_NUM_THREADS`` and
already multi-threads the serial path (110.6s single-threaded vs 63.5s). Workers
must therefore be pinned to one thread each or they oversubscribe; ``workers>1``
sets ``VECLIB_MAXIMUM_THREADS=1`` in the parent so spawned workers inherit it.
"""

import logging
import os
import re
import time
from collections import deque
from contextlib import contextmanager

import numpy as np
import pandas as pd

logging.getLogger('gensim').setLevel(logging.WARNING)

K_RANGE = range(2, 11)
N_RUNS = 50
ITERATIONS = 400
GAMMA_THRESHOLD = 0.001

#: Legacy fixed-pass budget. Passing ``passes=PASSES`` reproduces every ensemble
#: fitted before the early stop existed; it is not the default any more.
PASSES = 100

#: The pass budget is a **ceiling, not a cost**: the loop in :func:`_fit_batch`
#: stops as soon as the run settles, so over-provisioning buys headroom rather
#: than time. Measured on HRD: k=10 settles by pass ~64, k=5 around 150-200.
MAX_PASSES = 1000

#: ``topic diff`` must fall below this fraction of its own peak, for
#: :data:`PATIENCE` consecutive passes, before a fit counts as settled. Same
#: value the StripeSankey sweep uses for gensim, so the two stay comparable.
REL_TOL = 1e-3
PATIENCE = 2

#: Window for the opt-in ``move_tol`` rule: a fit is settled once no topic has
#: moved more than ``move_tol`` (Hellinger) over the last this-many passes.
MOVE_WINDOW = 100

#: gensim's own ``offset`` default. The pass loop's rho correction is stated
#: relative to it, so it is pinned here rather than left implicit.
_OFFSET = 1.0


def _as_krange(k_range):
    """Accept ``[lo, hi]`` (inclusive, as ``ensemble`` does) or any iterable."""
    if isinstance(k_range, (list, tuple)) and len(k_range) == 2:
        return range(int(k_range[0]), int(k_range[1]) + 1)
    return k_range


class _GensimLog(logging.Handler):
    """Capture the E-step convergence count gensim computes but never exposes.

    ``%i/%i documents converged within %i iterations``   (DEBUG, per chunk)
        Did the per-document E-step finish inside ``iterations``?

    gensim only reports it at DEBUG, so the logger has to be raised — and
    ``propagate`` turned off with it, or 450 models' worth of gensim INFO
    chatter lands in whatever logging the caller configured. :func:`_capture`
    does both and puts them back.

    gensim also logs ``topic diff`` per M-step; that is deliberately *not* used
    any more — see the module docstring, and :func:`_pass_diff` for what
    replaced it.
    """

    _DOCS = re.compile(r"(\d+)/(\d+) documents converged within (\d+) iterations")

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.docs_converged = []

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return
        m = self._DOCS.match(msg)
        if m:
            self.docs_converged.append((int(m.group(1)), int(m.group(2))))


def _pass_diff(before, after):
    """``topic diff`` over one pass: mean absolute change in ``Elogbeta``.

    Computed exactly as gensim's ``do_mstep`` computes the value it logs — same
    function, same ravelled float arrays — and then **rounded the way the log
    rounds it** (``%f``, six decimals). With one chunk per pass this is
    therefore the very number the early stop used to scrape from the log, so a
    cache fitted that way (HRD ``lda_ensemble_refit``) still reproduces down to
    the stopping pass. Without the rounding a borderline pass could tip.
    """
    from gensim.matutils import mean_absolute_difference

    return float("%f" % mean_absolute_difference(before.ravel(), after.ravel()))


def _hellinger_move(before, after):
    """Largest Hellinger distance any one topic moved over the pass.

    A diagnostic, not the stopping rule: it is in the unit the topics are
    clustered in, so a recorded ``hellinger_move_final`` says directly how far
    the pooled topics could still have been from where they would settle.
    """
    bc = (np.sqrt(before) * np.sqrt(after)).sum(axis=1)
    return float(np.sqrt(np.clip(1.0 - bc, 0.0, None)).max())


def _verdict(topic_diff, move, docs_converged, rel_tol, window_move=None,
             move_tol=None):
    """One fit's convergence record.

    The test is the per-pass ``topic diff`` against **this run's own peak**. An
    absolute tolerance is not portable — ``topic diff`` is a mean absolute change
    in ``Elogbeta``, so its magnitude follows the vocabulary size — and a
    relative-*decrement* rule would fail every run, because the series decays as
    a power law and never truly plateaus (measured tail exponent on HRD at k=5:
    -1.79).

    ``docs_converged`` holds the final pass's chunks; they are summed, so a
    multi-chunk fit reports the whole corpus rather than its last chunk.

    Under the ``move_tol`` rule, ``converged`` is that rule's test instead:
    ``window_move`` (largest topic movement over the last ``move_window``
    passes, NaN if the fit ran fewer) below ``move_tol``.
    """
    td = np.asarray(topic_diff, dtype=float)
    finite = td[np.isfinite(td)] if td.size else td
    out = {"passes_run": int(td.size)}
    if finite.size:
        peak = float(finite.max())
        ratio = float(finite[-1] / peak) if peak > 0 else 0.0
        out["topic_diff_final"] = float(finite[-1])
        out["topic_diff_ratio"] = ratio
        out["converged"] = bool(ratio < rel_tol)
    else:
        out["topic_diff_final"] = float("nan")
        out["topic_diff_ratio"] = float("nan")
        out["converged"] = None
    out["hellinger_move_final"] = float(move[-1]) if move else float("nan")
    out["window_move_final"] = (float("nan") if window_move is None
                                else float(window_move))
    if move_tol is not None:
        out["converged"] = bool(window_move is not None
                                and window_move < move_tol)
    got = sum(g for g, _ in docs_converged)
    total = sum(t for _, t in docs_converged)
    out["docs_converged_frac"] = got / total if total else float("nan")
    return out


@contextmanager
def _capture():
    """Install a :class:`_GensimLog` on gensim's LDA logger for the fit."""
    lg = logging.getLogger("gensim.models.ldamodel")
    handler, level, propagate = _GensimLog(), lg.level, lg.propagate
    lg.addHandler(handler)
    lg.setLevel(logging.DEBUG)
    lg.propagate = False
    try:
        yield handler
    finally:
        lg.removeHandler(handler)
        lg.setLevel(level)
        lg.propagate = propagate


def _settled(topic_diff, rel_tol, patience=PATIENCE):
    """Has ``topic diff`` stayed under ``rel_tol`` of this run's peak?

    ``patience`` consecutive passes must agree, so a single lucky pass does not
    end training early.
    """
    finite = [x for x in topic_diff if np.isfinite(x)]
    if len(finite) < patience + 1:
        return False
    peak = max(finite)
    if not peak > 0:
        return False
    return all(x / peak < rel_tol for x in finite[-patience:])


def _window_move(history, window):
    """Largest topic movement between the oldest and newest of ``window + 1``
    pass-end snapshots, or ``None`` until that many exist."""
    if len(history) < window + 1:
        return None
    return _hellinger_move(history[0], history[-1])


def _run_pass(model, corpus, chunksize, pass_):
    """Run pass ``pass_`` of a native multi-pass fit as its own ``update()`` call.

    gensim has no early stop: ``passes`` runs exactly ``passes`` times whether or
    not the model settled 60 passes ago. So the pass loop is driven from outside,
    one ``update(passes=1)`` per pass. That is only bit-identical to a native
    multi-pass fit because three gensim internals are compensated for
    (:func:`verify_pass_loop` is the test that says so, and the test is the
    point — all three are undocumented):

    1. ``rho = (offset + pass_ + num_updates/chunksize) ** -decay``. A native run
       calls ``do_mstep(rho, other, extra_pass=(pass_ > 0))``, so ``num_updates``
       grows chunk by chunk **on pass 0 only** and then freezes at the corpus
       size, leaving rho constant within each later pass. Inside a one-pass
       ``update()`` ``pass_`` is always 0, so every chunk would count again. The
       fix is to do what native does: from pass 1 on, ``do_mstep`` is called
       with ``extra_pass=True`` (``num_updates`` stays frozen) and the missing
       ``pass_`` term is moved into ``offset``. This is exact for any number of
       chunks — the earlier "``offset + 1`` and let it grow" correction was only
       right for one chunk, which is why the loop used to refuse SO-CPR.
    2. ``update()`` does ``state.numdocs += lencorpus`` and ``blend()`` uses
       ``state.numdocs`` as the target size that scales the sufficient
       statistics. Left alone it grows by a corpus per pass and reweights every
       blend, so it is reset before each call.
    3. gensim clamps ``chunksize = min(lencorpus, self.chunksize)`` **only when**
       ``update()`` receives ``chunksize=None``. The loop passes it explicitly,
       so it must pass that same effective value or (1)'s arithmetic is wrong.

    Skipping the compensations is not a rounding error: an uncorrected loop
    measured ``topic diff`` deviating by 7.4e-03 and topics by 2.1e-03 from the
    native fit within a dozen passes.
    """
    model.state.numdocs = 0                                            # (2)
    if pass_ == 0:
        model.update(corpus, passes=1, chunksize=chunksize, offset=_OFFSET)
        return
    native = type(model).do_mstep
    model.do_mstep = (lambda rho, other, extra_pass=False:
                      native(model, rho, other, True))                 # (1)
    try:
        model.update(corpus, passes=1, chunksize=chunksize,
                     offset=_OFFSET + pass_)                           # (1)
    finally:
        del model.do_mstep


def _fit_batch(jobs, corpus, id2word, params):
    """Fit every ``(k, run, seed)`` in ``jobs``; return pooled per-job results.

    One call per worker, not per fit — see the module docstring on batching.

    Every fit — fixed ``passes`` or early stop — goes through the same pass loop
    (:func:`_run_pass`), so a fixed-pass fit gets the same per-pass convergence
    record instead of gensim's per-chunk one. The model is built with
    ``corpus=None`` so its initial state can be read before pass 0; the first
    ``topic diff`` (initial state -> end of pass 0) is the peak the stopping
    rule is relative to, as it always was.
    """
    from gensim.models import LdaModel

    n_docs = len(corpus)
    chunksize = min(n_docs, params["chunksize"])          # (3)
    n_chunks = -(-n_docs // chunksize)
    fixed = params["passes"]
    budget = fixed if fixed is not None else params["max_passes"]
    move_tol, window = params.get("move_tol"), params.get("move_window", MOVE_WINDOW)
    out = []
    for k, run, seed in jobs:
        t_fit = time.time()
        model = LdaModel(id2word=id2word, num_topics=k, random_state=seed,
                         eval_every=None, chunksize=chunksize, offset=_OFFSET,
                         iterations=params["iterations"],
                         gamma_threshold=params["gamma_threshold"],
                         alpha=params["alpha"], eta=params["eta"])
        elog, topics = model.state.get_Elogbeta(), model.get_topics()
        topic_diff, move = [], []
        # Only the window rule needs past topics, and only `move_window + 1` of
        # them; HRD-sized vocabularies would make keeping them by default costly.
        history = (deque([topics], maxlen=window + 1) if move_tol is not None
                   else None)
        with _capture() as cap:
            for pass_ in range(budget):
                _run_pass(model, corpus, chunksize, pass_)
                elog_new, topics_new = model.state.get_Elogbeta(), model.get_topics()
                topic_diff.append(_pass_diff(elog, elog_new))
                move.append(_hellinger_move(topics, topics_new))
                elog, topics = elog_new, topics_new
                if history is not None:
                    history.append(topics)
                if fixed is not None:
                    continue
                if history is not None:
                    wm = _window_move(history, window)
                    if wm is not None and wm < move_tol:
                        break
                elif _settled(topic_diff, params["rel_tol"]):
                    break
        diag = _verdict(topic_diff, move, cap.docs_converged[-n_chunks:],
                        params["rel_tol"],
                        window_move=(None if history is None
                                     else _window_move(history, window)),
                        move_tol=move_tol)
        # inference() runs the E-step over the whole corpus in one call. The old
        # per-document get_document_topics() loop is equivalent but re-draws the
        # initial gamma per document; with the E-step actually converging the two
        # agree, and this is the cheaper, more direct route.
        gamma, _ = model.inference(corpus)
        doc_topic = gamma / gamma.sum(axis=1, keepdims=True)
        if params.get("log_fits"):
            # Workers report per fit: joblib only reports a batch when the whole
            # batch is done, which for an SO-CPR sweep is the whole run.
            print(f"  fit k={k} run={run}: {diag['passes_run']} passes, "
                  f"{'settled' if diag['converged'] else 'NOT settled'}, "
                  f"{time.time() - t_fit:.0f}s  [{time.strftime('%H:%M:%S')}]",
                  flush=True)
        out.append((k, run, model.get_topics().astype(np.float32),
                    doc_topic.astype(np.float32), diag))
    return out


def _report_convergence(diags, *, rel_tol, min_converged, max_passes,
                        iterations, fixed_passes=None, move_tol=None,
                        move_window=MOVE_WINDOW, verbose=True):
    """Summarise the ensemble's convergence, and warn on whatever failed.

    Nothing is refused and no fit is dropped. An unsettled fit is one topic
    source among hundreds, and silently discarding it would bias the pool
    towards the k values that happen to settle fastest — which is exactly the
    structure the clustering is supposed to find, not assume. So it is counted,
    named and left in.
    """
    import warnings

    n = len(diags)
    settled = diags["converged"].fillna(False).astype(bool)
    poor = diags["docs_converged_frac"] < min_converged

    if verbose:
        pr, ratio = diags["passes_run"], diags["topic_diff_ratio"]
        frac = diags["docs_converged_frac"]
        rule = (f"move_tol={move_tol:g} over {move_window} passes"
                if move_tol is not None else f"rel_tol={rel_tol:g}")
        print(f"Convergence ({rule}"
              + (f", passes={fixed_passes} fixed" if fixed_passes is not None
                 else f", max_passes={max_passes}")
              + f"): {int(settled.sum())}/{n} fits settled", flush=True)
        print(f"  passes run      {pr.min()}-{pr.max()} (median {pr.median():.0f})",
              flush=True)
        print(f"  topic_diff/peak {ratio.min():.1e}-{ratio.max():.1e} "
              f"(median {ratio.median():.1e})", flush=True)
        hm = diags["hellinger_move_final"]
        print(f"  last-pass move  {hm.min():.1e}-{hm.max():.1e} Hellinger "
              f"(median {hm.median():.1e}, largest single topic)", flush=True)
        wm = diags["window_move_final"]
        if wm.notna().any():
            print(f"  {move_window}-pass move {wm.min():.1e}-{wm.max():.1e} "
                  f"Hellinger (median {wm.median():.1e})", flush=True)
        print(f"  E-step docs     {frac.min():.3f}-{frac.max():.3f} converged "
              f"within iterations={iterations}", flush=True)
        if not settled.all():
            by_k = diags.loc[~settled].groupby("k").size()
            print("  NOT settled: "
                  + ", ".join(f"k={k} ({c} fits)" for k, c in by_k.items()),
                  flush=True)

    if not settled.all() and move_tol is not None:
        worst = diags.loc[diags["window_move_final"].fillna(np.inf).idxmax()]
        warnings.warn(
            f"{int((~settled).sum())}/{n} fits did not settle: some topic still "
            f"moved {move_tol:g} Hellinger or more over the last {move_window} "
            f"passes (worst: k={int(worst['k'])} run={int(worst['run'])} at "
            f"{worst['window_move_final']:.3f} after {int(worst['passes_run'])} "
            "passes). "
            + ("Raise `max_passes`." if fixed_passes is None else
               "Leave `passes=None` to let each fit stop when it settles.")
            + " Pooled topics from a model still moving add spread that the "
            "clustering will read as structure.",
            RuntimeWarning, stacklevel=3,
        )
    elif not settled.all():
        worst = diags.loc[diags["topic_diff_ratio"].idxmax()]
        warnings.warn(
            f"{int((~settled).sum())}/{n} fits did not settle: topic diff never "
            f"fell below {rel_tol:g} of its own peak (worst: k={int(worst['k'])} "
            f"run={int(worst['run'])} at {worst['topic_diff_ratio']:.2e} after "
            f"{int(worst['passes_run'])} passes). "
            + ("Raise `max_passes`." if fixed_passes is None else
               "Leave `passes=None` to let each fit stop when it settles.")
            + " Pooled topics from a model still moving add spread that the "
            "clustering will read as structure.",
            RuntimeWarning, stacklevel=3,
        )
    if poor.any():
        w = diags.loc[diags["docs_converged_frac"].idxmin()]
        warnings.warn(
            f"{int(poor.sum())}/{n} fits ended with under {min_converged:.0%} of "
            f"documents converged in the E-step (worst: k={int(w['k'])} "
            f"run={int(w['run'])} at {w['docs_converged_frac']:.0%}). Raise "
            f"`iterations` above {iterations}.",
            RuntimeWarning, stacklevel=3,
        )
    return diags


def run_lda_ensemble(V, feature_names, k_range=K_RANGE, n_runs=N_RUNS,
                     passes=None, max_passes=MAX_PASSES, rel_tol=REL_TOL,
                     move_tol=None, move_window=MOVE_WINDOW,
                     iterations=ITERATIONS,
                     gamma_threshold=GAMMA_THRESHOLD, alpha="symmetric", eta=None,
                     chunksize=2000, seed_base=0, workers=1, progress=True,
                     return_doc_topic=False, min_converged=0.8):
    """Fit a gensim LDA ensemble under variational Bayes and pool every topic.

    For each ``k`` and each of ``n_runs`` seeds, an ``LdaModel`` is fitted and
    every one of its ``k`` topic-word distributions is stacked into a single
    ``topic_feature_df``. ``topic_meta_df`` records the ``(k, run, topic)`` each
    row came from plus that topic's ``weight`` (its total loading mass across
    samples), matching :func:`nmf.run_nmf_ensemble` exactly.

    Gensim's ``get_topics()`` already returns rows summing to 1, so the pooled
    topics are distributions without further normalisation (the NMF twin has to
    L1-normalise; ``clustering.hellinger_embed`` renormalises regardless).

    Parameters
    ----------
    V : array-like (n_samples, n_features)
        Non-negative table, samples are rows. **Counts** are what LDA's
        multinomial generative model assumes; relative abundances will fit
        without complaint but the generative reading no longer holds.
    feature_names : sequence of str
        Column labels, length ``n_features``.
    k_range : ``[lo, hi]`` (inclusive) or iterable of int
        Topic counts to sweep.
    n_runs : int
        Seeds per k. ``random_state = seed_base + k*1000 + run``, so the whole
        ensemble is reproducible.
    passes : int, optional
        Fit exactly this many passes with no early stop — what a native gensim
        fit does (it still goes through the pass loop, bit-identically, so it
        gets a per-pass convergence record). Leave at ``None`` (the default) to
        stop each fit when it settles. ``passes=100`` reproduces every ensemble
        cached before the early stop existed.
    max_passes : int
        Ceiling on the early-stop loop. A ceiling, not a cost: a fit that
        settles at pass 64 stops at 64.
    rel_tol : float
        A fit is settled once ``topic diff`` has sat below this fraction of its
        own peak for :data:`PATIENCE` passes. See :func:`_verdict` for why the
        test is relative rather than absolute.
    move_tol, move_window : float or None, int
        Opt-in stopping rule that replaces ``rel_tol``: stop once no topic has
        moved more than ``move_tol`` (Hellinger) over the last ``move_window``
        passes. Use it where ``rel_tol`` is known to stop early — see the module
        docstring for SO-CPR, where 0.01 over 100 passes was what tracked the
        topics. ``None`` (default) keeps ``rel_tol``.
    iterations, gamma_threshold, alpha, eta, chunksize
        Passed to ``LdaModel``. See the module docstring for why the defaults
        are what they are — in particular ``iterations=400`` is a correctness
        fix, not tuning. ``chunksize`` is only ever used as
        ``min(n_samples, chunksize)``, which is gensim's own clamp. A table with
        more samples than ``chunksize`` is fitted online (one M-step per chunk);
        the stopping rule still judges whole passes.
    workers : int
        Independent fits to run in parallel processes. 1 = serial.
    return_doc_topic : bool
        Also return the pooled sample loadings (see Returns).
    min_converged : float
        Warn if a fit ends with a smaller fraction of documents having converged
        in the E-step. This is the LDA counterpart of sklearn's
        ``ConvergenceWarning`` on the NMF side, which gensim does not raise.

    Returns
    -------
    topic_feature_df : pd.DataFrame (n_topics_total, n_features)
        One row per pooled topic; columns are ``feature_names``.
    topic_meta_df : pd.DataFrame
        Columns ``k, run, topic, weight``, aligned row-for-row, plus that fit's
        convergence record repeated on each of its topics: ``passes_run``,
        ``topic_diff_final``, ``topic_diff_ratio``, ``converged``,
        ``hellinger_move_final``, ``window_move_final``, ``docs_converged_frac``. It rides along in the meta frame so it survives
        ``ensemble.save_ensemble`` — a cached ensemble can be asked, later,
        whether it converged.
    doc_topic : np.ndarray (n_topics_total, n_samples), only if
        ``return_doc_topic`` — each topic's loading across samples, i.e. the
        pooled H laid out topic-major so it stays aligned with the other two.
    """
    V = np.asarray(V, dtype=np.float64)
    if np.any(V < 0):
        raise ValueError("V must be non-negative.")
    feature_names = list(feature_names)
    if V.shape[1] != len(feature_names):
        raise ValueError(
            f"feature_names length {len(feature_names)} != V.shape[1] {V.shape[1]}"
        )

    from gensim.matutils import Dense2Corpus

    corpus = list(Dense2Corpus(V, documents_columns=False))
    id2word = dict(enumerate(feature_names))
    params = dict(passes=passes, max_passes=max_passes, rel_tol=rel_tol,
                  move_tol=move_tol, move_window=move_window,
                  log_fits=bool(progress and workers and workers > 1),
                  iterations=iterations, gamma_threshold=gamma_threshold,
                  alpha=alpha, eta=eta, chunksize=chunksize)

    jobs = [(k, run, seed_base + k * 1000 + run)
            for k in _as_krange(k_range) for run in range(n_runs)]
    t0 = time.time()

    if workers and workers > 1:
        from joblib import Parallel, delayed

        # Accelerate ignores OMP_NUM_THREADS; workers inherit the parent's env at
        # spawn, so this has to be set here rather than inside _fit_batch (where
        # numpy is already initialised and the setting would be ignored).
        saved = {k: os.environ.get(k)
                 for k in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS")}
        os.environ.update({k: "1" for k in saved})
        try:
            batches = [jobs[i::workers] for i in range(workers)]
            results = Parallel(n_jobs=workers, backend="loky",
                               verbose=10 if progress else 0)(
                delayed(_fit_batch)(b, corpus, id2word, params) for b in batches if b
            )
        finally:
            for key, val in saved.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val
        fits = [f for batch in results for f in batch]
    else:
        fits = []
        for i, job in enumerate(jobs, 1):
            fits.extend(_fit_batch([job], corpus, id2word, params))
            if progress and (i == 1 or i % 10 == 0 or i == len(jobs)):
                per = (time.time() - t0) / i
                print(f"[{i}/{len(jobs)}] {time.time()-t0:.0f}s elapsed, "
                      f"~{(len(jobs)-i)*per/60:.1f}min remaining", flush=True)

    fits.sort(key=lambda f: (f[0], f[1]))

    diags = pd.DataFrame([{"k": k, "run": run, **d} for k, run, _, _, d in fits])
    _report_convergence(diags, rel_tol=rel_tol, min_converged=min_converged,
                        max_passes=max_passes, iterations=iterations,
                        fixed_passes=passes, move_tol=move_tol,
                        move_window=move_window, verbose=progress)

    topic_feature = np.vstack([f[2] for f in fits])
    doc_topic = np.hstack([f[3] for f in fits]).T   # (n_topics_total, n_samples)
    meta = pd.DataFrame(
        [{"k": k, "run": run, "topic": t, "weight": float(dt[:, t].sum())}
         for k, run, tw, dt, _ in fits for t in range(tw.shape[0])]
    ).merge(diags, on=["k", "run"], how="left")

    topic_feature_df = pd.DataFrame(topic_feature, columns=feature_names)
    if progress:
        print(f"Fitted {len(fits)} models -> {len(topic_feature_df)} topics "
              f"in {time.time() - t0:.0f}s", flush=True)
    if return_doc_topic:
        return topic_feature_df, meta, doc_topic
    return topic_feature_df, meta


def verify_pass_loop(V=None, feature_names=None, *, k=4, passes=12, seed=7,
                     chunksizes=(2000, 32), verbose=True):
    """Check that the pass loop still reproduces a native gensim fit exactly.

    :func:`_run_pass` drives gensim's pass loop from outside so a fit can stop
    when it settles, and compensates for three gensim internals to do so (the
    ``rho`` schedule, ``state.numdocs``, and the ``chunksize`` clamp). All three
    are undocumented, so this fits the same model natively —
    ``LdaModel(corpus, passes=passes)``, nothing of ours in the way — and through
    :func:`_fit_batch` both ways (fixed ``passes``, and the early-stop loop with a
    ``rel_tol`` it can never meet), and compares. It is what turns "we poke
    private attributes" into a tested invariant: if a gensim release changes
    any of them, this fails loudly instead of the ensemble silently becoming a
    different model.

    ``chunksizes`` are tried in turn. The default covers both regimes: 2000 puts
    the synthetic table in one chunk (HRD-like), 32 splits it into three
    (SO-CPR-like, online with several M-steps per pass).

    Pass a table, or leave it out for a small synthetic one.

    Returns a dict with ``ok`` and the measured deviations per chunksize.
    """
    from gensim.matutils import Dense2Corpus
    from gensim.models import LdaModel

    if V is None:
        rng = np.random.default_rng(0)
        V = rng.poisson(6, (90, 120))
        feature_names = [f"w{i}" for i in range(120)]
    V = np.asarray(V, dtype=np.float64)
    feature_names = list(feature_names)
    corpus = list(Dense2Corpus(V, documents_columns=False))
    id2word = dict(enumerate(feature_names))

    out = {"ok": True, "passes": passes}
    for chunksize in chunksizes:
        native = LdaModel(corpus=corpus, id2word=id2word, num_topics=k,
                          random_state=seed, eval_every=None, passes=passes,
                          chunksize=chunksize, offset=_OFFSET,
                          iterations=ITERATIONS, gamma_threshold=GAMMA_THRESHOLD,
                          alpha="symmetric", eta=None)
        want = native.get_topics().astype(np.float32)
        gamma, _ = native.inference(corpus)
        want_dt = (gamma / gamma.sum(axis=1, keepdims=True)).astype(np.float32)

        params = dict(passes=passes, max_passes=passes, rel_tol=0.0,
                      iterations=ITERATIONS, gamma_threshold=GAMMA_THRESHOLD,
                      alpha="symmetric", eta=None, chunksize=chunksize)
        # rel_tol=0 can never be met, so the early-stop loop runs all `max_passes`.
        for label, prm in (("fixed", params), ("early-stop", {**params, "passes": None})):
            _, _, topics, doc_topic, diag = _fit_batch([(k, 0, seed)], corpus,
                                                       id2word, prm)[0]
            dev = (float(np.abs(want - topics).max()),
                   float(np.abs(want_dt - doc_topic).max()))
            good = dev == (0.0, 0.0) and diag["passes_run"] == passes
            out["ok"] &= good
            n_chunks = -(-len(corpus) // min(chunksize, len(corpus)))
            out[(chunksize, label)] = {"topics_deviation": dev[0],
                                       "doc_topic_deviation": dev[1],
                                       "passes_run": diag["passes_run"],
                                       "chunks_per_pass": n_chunks}
            if verbose:
                print(f"[verify_pass_loop] {'OK' if good else 'MISMATCH'} — "
                      f"{n_chunks} chunk(s)/pass, {label}: topics {dev[0]:.3e}, "
                      f"doc-topics {dev[1]:.3e}, {diag['passes_run']}/{passes} "
                      "passes")
    return out


def main(data_dir="data", out_dir=None, workers=1):
    """CLI entry point: fit the ensemble on ``<data_dir>/asv_table.csv`` and cache it."""
    from ensemble import save_ensemble

    out_dir = out_dir or os.path.join(data_dir, "lda_ensemble_results")
    asv_df = pd.read_csv(os.path.join(data_dir, "asv_table.csv"), index_col=0)

    topic_feature_df, topic_meta_df, doc_topic = run_lda_ensemble(
        asv_df.values, asv_df.columns, workers=workers, return_doc_topic=True,
    )
    save_ensemble(out_dir, topic_feature_df, topic_meta_df,
                  doc_topic=doc_topic, sample_ids=list(asv_df.index))
    print(f"Saved to {out_dir}")


# --------------------------------------------------------------------------- #
# Fixed-factor inference under the Dirichlet prior
# --------------------------------------------------------------------------- #
def infer_loadings_fixed_topics_lda(V, topics, alpha=None, random_state=0):
    """NMF counterpart is :func:`nmf.infer_loadings_fixed_topics` — freeze W, solve H.

    Runs sklearn's variational E-step with the topics injected as
    ``components_``, i.e. ``LatentDirichletAllocation.transform`` against a fixed
    dictionary. Deterministic and reproducible: with W frozen the problem is
    convex. The Dirichlet ``doc_topic_prior`` (``alpha``) is what regularises H
    against a redundant medoid dictionary, and is the reason an LDA ensemble must
    *not* be folded in through the NMF-GKL path.

    ``components_`` are unnormalised variational pseudo-counts, not
    probabilities, so the topic distributions are rescaled by the corpus mass
    before injection.

    Parameters
    ----------
    V : array-like (n_samples, n_features)
        The count table the ensemble was fit on, or new samples with the same
        features in the same column order.
    topics : array-like (n_topics, n_features)
        Fixed topic-feature distributions (rows sum to 1).
    alpha : float, optional
        ``doc_topic_prior``. ``None`` gives sklearn's default ``1/n_topics``.

    Returns
    -------
    loadings : np.ndarray (n_samples, n_topics), rows sum to 1.
    """
    from scipy.special import psi
    from sklearn.decomposition import LatentDirichletAllocation

    V = np.asarray(V, dtype=float)
    topics = np.asarray(topics, dtype=float)
    if topics.shape[1] != V.shape[1]:
        raise ValueError(
            f"topics has {topics.shape[1]} features, V has {V.shape[1]}; "
            "they must share the same feature axis / column order."
        )
    k = topics.shape[0]

    lda = LatentDirichletAllocation(
        n_components=k, doc_topic_prior=alpha, learning_method="online",
        max_iter=1, random_state=random_state,
    )
    lda.fit(V)  # bootstraps the fitted attributes; the topics are replaced below
    lda.components_ = topics * (V.sum() / k) + 1e-6
    lda.exp_dirichlet_component_ = np.exp(
        psi(lda.components_) - psi(lda.components_.sum(axis=1))[:, None]
    )
    return lda.transform(V)


# --------------------------------------------------------------------------- #
# Guided-vs-Standard track primitives (paired with evaluate.py)
# --------------------------------------------------------------------------- #
def run_lda(count_matrix, feature_names, k, n_iter=100, seed=1):
    """Run standard LDA via gensim.
    count_matrix : array-like (n_samples, n_features), integer counts
    Returns dict with 'W' (n_features, k) and 'H' (k, n_samples) for compatibility with NMF functions.
    """
    from gensim.models import LdaModel as _LdaModel

    count_arr = np.asarray(count_matrix)
    n_samples, n_features = count_arr.shape
    feature_names = [str(f) for f in feature_names]
    id2word = dict(enumerate(feature_names))
    corpus = []
    for i in range(n_samples):
        doc = [(j, int(count_arr[i, j])) for j in range(n_features) if count_arr[i, j] > 0]
        corpus.append(doc)

    model = _LdaModel(
        corpus=corpus, id2word=id2word, num_topics=k,
        passes=n_iter, random_state=seed, alpha='auto', eta='auto',
    )

    W = np.zeros((n_features, k))
    for t in range(k):
        for word_id, prob in model.get_topic_terms(t, topn=n_features):
            W[word_id, t] = prob

    H = np.zeros((k, n_samples))
    for i, doc_bow in enumerate(corpus):
        for t, prob in model.get_document_topics(doc_bow, minimum_probability=0):
            H[t, i] = prob

    return {'W': W, 'H': H, 'model': model}


def print_lda_topics(topic_word, feature_names, n_top=10, display_names=None):
    """Print LDA topics from a topic-word matrix (guidedlda or gensim).
    topic_word : (k, n_features) — each row sums to ~1
    If display_names is provided, aggregate by display name."""
    k = topic_word.shape[0]
    topic_share = topic_word.sum(axis=1)
    topic_share = topic_share / (topic_share.sum() + 1e-12)
    if display_names is not None:
        show = np.asarray(display_names)
        for i in range(k):
            genus_weights = {}
            for j in range(topic_word.shape[1]):
                g = show[j]
                genus_weights[g] = genus_weights.get(g, 0) + topic_word[i, j]
            ranked = sorted(genus_weights.items(), key=lambda x: x[1], reverse=True)[:n_top]
            parts = [f"{name} ({w:.4f})" for name, w in ranked]
            print(f"  Topic {i+1}: {', '.join(parts)}")
    else:
        for i in range(k):
            top_idx = topic_word[i].argsort()[-n_top:][::-1]
            weights = topic_word[i, top_idx]
            parts = [f"{feature_names[j]} ({weights[idx]:.4f})" for idx, j in enumerate(top_idx)]
            print(f"  Topic {i+1}: {', '.join(parts)}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    main(data_dir=a.data_dir, out_dir=a.out_dir, workers=a.workers)
