"""foreist -- SimulacraBench submission.

Each blank cell is scored on its own, so the best vector for a cell is the
item's distribution conditional on what is known about that respondent: the
GIVEN columns. This file estimates P(answer | GIVEN) item by item, at runtime,
from the visible rows predict() receives -- the real instrument is visible
nowhere else, so every choice below is made by cross-validation inside predict.

The gate sentinel is treated as one more answer. Whether a respondent was asked
is learned from the rows rather than read off the routing, which the UNHCR file
does not obey (gate agreement 0.29-0.98) and which a GIVEN parent then reveals
with whatever reliability the data actually has.

Per item, several estimators, each scored out of fold (K random folds over the
visible rows, the grader's floor applied):

  chain    Dirichlet backoff over GIVEN columns, chosen greedily with every
           alpha_l by exact leave-one-out log score:
               p_l = (counts in cell_l + alpha_l p_{l-1}) / (n in cell_l + alpha_l)
           A level is kept only if its LOO gain clears two standard errors,
           and the out-of-fold rows of fold k come from a chain chosen without
           fold k, so its OOF score is not flattered by its own selection.
  linear   multinomial logit on one-hot GIVEN, all items in one multi-task fit
  mlp      one shared hidden layer, a softmax head per item

and, for a gated item, the same vector built in two parts,
P(asked) * P(answer | asked) next to 1 - P(asked):

  split    a chain for "asked" over every row, a chain for the answer over
           the asked rows only, each free to pick its own columns
  gate     P(asked) = sum_v P(parent = v) T[v], with the parent's fold-excluded
           chain prediction and T[v] the rate at which the child was asked when the
           parent was v -- counted, so a gate the rows do not obey is learned
  official the organisers' external_model estimator without its LLM: two
           MI-ranked GIVEN strata at alpha 2, the gate recursion down the
           chain (a GIVEN parent by its counted rate, never 0/1)

plus the item's marginal as a floor under everything. Items are stacked
parents first; gate components use the independently fitted parent chain.

The two nets are trained for every fold and on all rows at once, with each
item's epoch picked at the checkpoint where its own out-of-fold score peaks.
Per item the estimators are then mixed with simplex weights and a temperature
that maximise the out-of-fold score; the final vector applies those to
estimators refitted on every visible row.
"""

import json
import os
import time

_IMPORTED = time.monotonic()
for _key, _value in (("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"),
                     ("HF_HUB_DISABLE_TELEMETRY", "1")):
    os.environ.setdefault(_key, _value)

import numpy as np
import pandas as pd

FLOOR = 1e-3                      # scoring.floor in config.yml
BUDGET = {1: 900.0, 2: 3600.0}    # seconds, all three instruments together
CALLS = 3
# PREDICT items x frame rows summed over the three instruments, per phase
# (schema counts: UNICEF 12 items, World Bank 76, UNHCR 213). A call's share
# of the clock follows its share of this work, not one third each.
WORK = {1: 12 * 16373 + 76 * 5715 + 213 * 8900,
        2: 12 * 19846 + 76 * 6927 + 213 * 10349}
SEED = 20261030
FOLDS = 5

ALPHAS = np.geomspace(0.05, 2048.0, 24)
MAX_DEPTH = 12                    # a level is still kept only if its LOO gain clears 2 SE
MIN_GAIN = 2e-4                   # nats per visible row a new level must buy
CHAIN_SE = 2.0                    # ... and this many standard errors of that gain
CHAIN_CANDIDATES = 6              # columns per depth that get the exact LOO search
CHAIN_MIN_ROWS = 5                # fewer labelled rows than this: the marginal

GATE_BETA = 8.0                  # fixed pseudo-rows per parent value
OFFICIAL_ALPHA = 2.0              # baseline/external_model, as shipped
OFFICIAL_STRATA = 2
OFFICIAL_MIN_COUNT = 8

NET_MIN_ROWS = 200
LINEAR_STEPS = (10, 20, 35, 55, 80, 110, 150, 200, 260, 330, 420, 540, 700)
MLP_STEPS = (8, 16, 25, 35, 50, 70, 95, 125, 160, 200, 250)
NETS = (
    {"name": "linear", "hidden": 0, "lr": 0.05, "wd": 1e-4, "dropout": 0.0,
     "steps": LINEAR_STEPS},
    {"name": "mlp", "hidden": 128, "lr": 0.004, "wd": 1e-3, "dropout": 0.25,
     "steps": MLP_STEPS},
    # One-hot GIVEN plus pairs selected from their cardinalities only, so
    # feature selection cannot read a validation fold's answers.
    {"name": "mlp_pairs", "hidden": 128, "lr": 0.004, "wd": 1e-3,
     "dropout": 0.25, "seed": 2, "steps": MLP_STEPS, "pairs": True},
    {"name": "mlp_split", "hidden": 128, "lr": 0.004, "wd": 1e-3,
     "dropout": 0.25, "seed": 3, "steps": MLP_STEPS, "split": True},
    {"name": "mlp_wide", "hidden": 384, "lr": 0.002, "wd": 3e-3, "dropout": 0.4,
     "seed": 1, "steps": MLP_STEPS},
)
PAIR_COLUMNS = 8                  # lowest-cardinality nonconstant GIVEN columns
EM_STEPS = 200
EM_TOL = 1e-4
TEMPERATURES = (0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4)
RECAL_SHARE = 0.2                 # visible rows held out of a whole inner run
RECAL_TIME = 0.4                  # share of the call's clock the inner run may use
# A global temperature / marginal mix on top bought nothing on the development
# leaderboard and blurs cells a GIVEN gate decides, so both grids are identity.
RECAL_T = (1.0,)
RECAL_MIX = (0.0,)
RECAL_MIN_GAIN = 1e-4             # nats per cell the recalibration must buy
# Pretrained tabular models mixed into the stack: name -> (library, bundled
# checkpoint, ensemble members). Checkpoints are unmodified public files:
#   Prior-Labs/tabpfn_3 @ 24a16a89d245, jingang/TabICL @ 4dcd344ece2c,
#   Prior-Labs/tabpfn_3_5 @ 06bf2ba35c80.
FM_SPECS = {
    "tabpfn35": ("tabpfn", "tabpfn-v3.5-fast-20260909.safetensors", 4),
    "tabicl": ("tabicl", "tabicl-classifier-v2-20260212.ckpt", 1),
    "tabpfn": ("tabpfn", "tabpfn-v3-classifier-v3_default.ckpt", 2),
}
FM_NAMES = ("tabpfn35", "tabicl", "tabpfn")
FM_W = tuple(round(0.1 * i, 1) for i in range(11))
FM_SHRINK = 5000.0                # held-out cells' worth of the instrument-wide curve
                                  # behind each item's own choice of weights

_calls_done = [0]
_work_done = [0]
LAST_RUN = {}                     # diagnostics for the local harness only

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_prior():
    """The baked LLM option shares (bake_llm_prior.py), or None."""
    try:
        with open(os.path.join(HERE, "llm_prior.json"), encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


LLM_PRIOR = _load_prior()


# ------------------------------------------------------------------ helpers --

def _given_codes(column, values):
    """Codes for a GIVEN column: listed values, then NA_GATED, then anything else."""
    lookup = {value: i for i, value in enumerate(values)}
    lookup["NA_GATED"] = len(values)
    return (column.map(lookup).fillna(len(values) + 1)
            .to_numpy(dtype=np.int64)), len(values) + 2


def _target_codes(column, values, gated):
    """Codes for a PREDICT column, sentinel last; -1 for a blank or a stray."""
    lookup = {value: i for i, value in enumerate(values)}
    if gated:
        lookup["NA_GATED"] = len(values)
    return column.map(lookup).fillna(-1).to_numpy(dtype=np.int64)


def _floored(p, width):
    """What the grader does to a probability before taking its log."""
    return FLOOR + (1.0 - width * FLOOR) * p


def _log_score(p_true, width):
    return float(np.log(_floored(p_true, width)).mean())


def _marginal(y, width):
    counts = np.bincount(y, minlength=width).astype(float) + 0.5
    return counts / counts.sum()


def _items(schema):
    items = [name for name, record in schema["items"].items()
             if record["class"] in ("GIVEN", "PREDICT")]
    given = [n for n in items if schema["items"][n]["class"] == "GIVEN"]
    targets = [n for n in items if schema["items"][n]["class"] == "PREDICT"]
    return given, targets


# ------------------------------------------------------------ backoff chain --

def _chain_select(x, y, width):
    """Columns and alphas of the chain, chosen greedily by exact LOO score.

    At each depth the columns are first ranked by plug-in conditional entropy
    (Miller-Madow corrected), and only the best few get the exact LOO search.
    """
    n = y.size
    coarse = list(range(1, ALPHAS.size, 3))

    def level(key, cells, loo_parent):
        # Rows that share a cell and an answer share every coarser cell too,
        # so their LOO values coincide: score once per (cell, answer) group.
        group = key * width + y
        joint = np.bincount(group, minlength=cells * width)
        present = np.flatnonzero(joint)
        slot = np.empty(cells * width, dtype=np.int64)
        slot[present] = np.arange(present.size)
        member = np.empty(present.size, dtype=np.int64)
        member[slot[group]] = np.arange(n)
        weight = joint[present].astype(float)
        hit = weight - 1.0                              # leave this row out
        total = np.bincount(key, minlength=cells)[present // width] - 1.0
        parent = loo_parent[member]

        def scores(index):
            a = ALPHAS[index][:, None]
            loo = (hit[None] + a * parent[None]) / (total[None] + a)
            return (np.log(_floored(loo, width)) * weight).sum(axis=1) / n, loo

        values, loo = scores(coarse)
        k = coarse[int(np.argmax(values))]
        near = [i for i in (k - 1, k + 1) if 0 <= i < ALPHAS.size]
        fine, fine_loo = scores(near)
        candidates = [(values.max(), k, loo[int(np.argmax(values))])]
        candidates += [(fine[i], near[i], fine_loo[i]) for i in range(len(near))]
        value, index, best_loo = max(candidates, key=lambda c: c[0])
        return float(value), ALPHAS[index], best_loo[slot[group]]

    def entropy(key, cells):
        joint = np.bincount(key * width + y, minlength=cells * width)
        present = np.flatnonzero(joint)
        count = joint[present].astype(float)
        rows = np.bincount(key, minlength=cells)
        plug = -(count * np.log(count / rows[present // width])).sum() / n
        return plug + (present.size - (rows > 0).sum()) * 0.5 / n

    codes = [x[:, column] for column in range(x.shape[1])]
    cards = [int(code.max()) + 1 if n else 1 for code in codes]
    distinct = [int((np.bincount(code) > 0).sum()) if n else 0
                for code in codes]

    key, cells = np.zeros(n, dtype=np.int64), 1
    score, alpha, loo = level(key, cells, np.full(n, 1.0 / width))
    columns, alphas = [None], [alpha]
    remaining = [c for c in range(x.shape[1]) if distinct[c] > 1]
    for _ in range(MAX_DEPTH):
        ranked = sorted(remaining, key=lambda c: entropy(
            key * cards[c] + codes[c], cells * cards[c]))[:CHAIN_CANDIDATES]
        best = None
        for column in ranked:
            candidate = key * cards[column] + codes[column]
            value, alpha, cand_loo = level(candidate, cells * cards[column],
                                           loo)
            if best is None or value > best[0]:
                best = (value, column, candidate, alpha, cand_loo)
        if best is None:
            break
        # The best of a few dozen candidates always gains a little by luck;
        # ask for the gain to stand clear of its own row-to-row noise.
        diff = (np.log(_floored(best[4], width))
                - np.log(_floored(loo, width)))
        noise = diff.std() / np.sqrt(max(n - 1, 1))
        if best[0] - score < max(MIN_GAIN, CHAIN_SE * noise):
            break
        score, column, key, alpha, loo = best
        key, cells = _dense(key, cells * cards[column])
        remaining.remove(column)
        columns.append(column)
        alphas.append(alpha)
    return columns, alphas


def _dense(key, size):
    """Renumber cell ids 0.. in order, dropping empty ones: (key, cells)."""
    used = np.bincount(key, minlength=size) > 0
    renumber = np.cumsum(used) - 1
    return renumber[key], int(used.sum())


def _chain_keys(x_all, columns):
    """One cell id per row per level, over fit rows and query rows together."""
    key, cells = np.zeros(len(x_all), dtype=np.int64), 1
    keys = [key]
    for column in columns[1:]:
        card = int(x_all[:, column].max()) + 1
        key, cells = _dense(key * card + x_all[:, column], cells * card)
        keys.append(key)
    return keys


def _chain_probs(keys, alphas, y, width, n_fit):
    """Run the recursion for the rows after n_fit, counting the first n_fit."""
    p = np.full((len(keys[0]) - n_fit, width), 1.0 / width)
    for key, alpha in zip(keys, alphas):
        cells = int(key.max()) + 1
        joint = np.bincount(key[:n_fit] * width + y, minlength=cells * width)
        counts = joint.reshape(cells, width)[key[n_fit:]].astype(float)
        p = (counts + alpha * p) / (counts.sum(axis=1, keepdims=True) + alpha)
    return p


def _chain_predict(x_train, y_train, x_query, width):
    """Select a chain on the training rows, then predict the query rows."""
    if y_train.size < CHAIN_MIN_ROWS:
        return np.tile(_marginal(y_train, width), (len(x_query), 1)), [None]
    columns, alphas = _chain_select(x_train, y_train, width)
    keys = _chain_keys(np.concatenate([x_train, x_query]), columns)
    return _chain_probs(keys, alphas, y_train, width, y_train.size), columns


def _chain(x_fit, x_out, y, width, fold):
    """One target's chain: (oof (n, width), out (n_out, width), columns).

    y holds a code per fit row, -1 where the row does not count for this
    target; every fit row still gets an out-of-fold vector. Fold k's vectors
    come from a chain selected and counted on the other folds only.
    """
    labelled = y >= 0
    oof = np.empty((y.size, width))
    for k in range(FOLDS):
        train, query = labelled & (fold != k), fold == k
        oof[query] = _chain_predict(x_fit[train], y[train], x_fit[query],
                                    width)[0]
    out, columns = _chain_predict(x_fit[labelled], y[labelled], x_out, width)
    return oof, out, columns


PROFILE_ALPHAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


def _profile_keys(x_all):
    """Cell ids of the full GIVEN profile, and of the profile with each one
    column left out. The backoff chain stops at a few columns; these are the
    levels at the far end of it, and which one (if any) helps is left to the
    out-of-fold score."""
    keys = []
    columns = x_all.shape[1]
    for drop in [None] + list(range(columns)):
        cols = [c for c in range(columns) if c != drop]
        _, key = np.unique(x_all[:, cols], axis=0, return_inverse=True)
        keys.append(key.ravel())
    return keys


def _profile(keys, n, y, width, fold, prior):
    """Counts in a (nearly) full-profile cell, shrunk to the chain.

        p = (counts in the cell + alpha * chain) / (n in cell + alpha)

    Out-of-fold rows count only the other folds and shrink to the chain's own
    out-of-fold vector. The cell definition and alpha are the pair that scores
    best out of fold; the component is dropped when none beats the chain.
    """
    known = np.flatnonzero(y >= 0)
    yk = y[known]
    prior_oof, prior_out = prior
    pick = np.arange(known.size)
    best = (_log_score(prior_oof[known, yk], width) + 1e-6, None, None)
    tables = {}
    for d, key in enumerate(keys):
        kf = key[:n]
        cells = int(key.max()) + 1
        if cells < 2:
            continue
        full = np.zeros((cells, width))
        np.add.at(full, (kf[known], yk), 1.0)
        own = np.zeros((FOLDS, cells, width))
        np.add.at(own, (fold[known], kf[known], yk), 1.0)
        rest = full[kf[known]] - own[fold[known], kf[known]]     # (known, width)
        rest_n = rest.sum(1, keepdims=True)
        for alpha in PROFILE_ALPHAS:
            p = (rest + alpha * prior_oof[known]) / (rest_n + alpha)
            value = _log_score(p[pick, yk], width)
            if value > best[0]:
                best = (value, d, alpha)
        tables[d] = (full, own)
    if best[1] is None:
        return None
    _, d, alpha = best
    key = keys[d]
    kf, ko = key[:n], key[n:]
    full, own = tables[d]
    rest = full[kf] - own[fold, kf]
    oof = (rest + alpha * prior_oof) / (rest.sum(1, keepdims=True) + alpha)
    counts = full[ko]
    out = (counts + alpha * prior_out) / (counts.sum(1, keepdims=True) + alpha)
    return oof, out


def _two_part(asked, answer):
    """[P(asked) * P(answer | asked), 1 - P(asked)], the sentinel last."""
    asked = np.clip(asked, 0.0, 1.0)[:, None]
    return np.concatenate([asked * answer, 1.0 - asked], axis=1)


# ------------------------------------------------------------ gate parents --

def _gate_asked(parent_oof, parent_out, parent_seen, asked, fold):
    """Fold-excluded transition rates applied to fold-excluded parent chains.

    The smoothing mass is fixed in advance, not selected on the same OOF
    labels being predicted. No context or neighbour is selected from labels.
    """
    V = parent_oof.shape[1]
    both = (parent_seen >= 0) & (asked >= 0)
    cell = fold[both] * V + parent_seen[both]
    hit = np.bincount(cell, weights=asked[both], minlength=FOLDS * V
                      ).reshape(FOLDS, V)
    seen = np.bincount(cell, minlength=FOLDS * V).reshape(FOLDS, V)
    hit = np.concatenate([hit.sum(0) - hit, hit.sum(0)[None]])
    seen = np.concatenate([seen.sum(0) - seen, seen.sum(0)[None]])
    base = (hit.sum(1) + 0.5) / (seen.sum(1) + 1.0)
    table = (hit + GATE_BETA * base[:, None]) / (seen + GATE_BETA)
    return ((parent_oof * table[fold]).sum(1),
            parent_out @ table[FOLDS])


# ---------------------------------------------------------------- official --

def _mutual_information(x, y, nx, ny):
    """Plug-in mutual information in nats (baseline/external_model)."""
    if x.size < OFFICIAL_MIN_COUNT:
        return 0.0
    joint = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny) / x.size
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (px @ py)[nz])).sum())


def _official(x_fit, x_out, cards, Y, widths, gated, parents, order, fold):
    """baseline/external_model's estimator with a uniform base measure.

    Answers: the GIVEN columns with the most mutual information, ALPHA-2
    backoff over the asked rows. Asked: the gate recursion down the chain, a
    PREDICT parent through its answer distribution as shipped, a GIVEN parent
    through the rate it actually opened the child at in the rows. Per item
    (oof (n, w), out (n_out, w)).
    """
    n, J = len(x_fit), len(widths)
    x_all = np.concatenate([x_fit, x_out])
    oof = [np.empty((n, w)) for w in widths]
    out = [None] * J
    for k in list(range(FOLDS)) + [None]:
        train = np.ones(n, bool) if k is None else fold != k
        xt = x_fit[train]
        answer, asked = {}, {}
        for j in order:
            K = widths[j] - gated[j]
            y = Y[train, j]
            ok = (y >= 0) & (y < K)
            yy, xa = y[ok], xt[ok]
            ladder = sorted(range(len(cards)), reverse=True,
                            key=lambda g: _mutual_information(
                                xa[:, g], yy, cards[g], K))[:OFFICIAL_STRATA]
            running = np.full((len(x_all), K), 1.0 / K)
            key_fit = np.zeros(yy.size, np.int64)
            key_all = np.zeros(len(x_all), np.int64)
            size = 1
            for column in [None] + ladder:
                if column is not None:
                    key_fit = key_fit * cards[column] + xa[:, column]
                    key_all = key_all * cards[column] + x_all[:, column]
                    size *= cards[column]
                table = np.bincount(key_fit * K + yy, minlength=size * K
                                    ).reshape(size, K).astype(float)
                counts = table[key_all]
                running = ((counts + OFFICIAL_ALPHA * running)
                           / (counts.sum(1, keepdims=True) + OFFICIAL_ALPHA))
            answer[j] = running
            if not gated[j]:
                asked[j] = np.ones(len(x_all))
                full = running
            else:
                kind, index, opens = parents[j] or ("given", None, [])
                if kind == "predict":
                    asked[j] = asked[index] * (answer[index][:, opens].sum(1)
                                               if opens else 0.0)
                else:
                    # A GIVEN parent opens the child at the rate the rows
                    # show; with no usable parent, at the overall rate.
                    seen = y >= 0
                    v = (xt[seen, index] if index is not None
                         else np.zeros(int(seen.sum()), np.int64))
                    size = cards[index] if index is not None else 1
                    hit = np.bincount(v, weights=(y[seen] != K).astype(float),
                                      minlength=size)
                    total = np.bincount(v, minlength=size)
                    base = (hit.sum() + 0.5) / (total.sum() + 1.0)
                    rate = (hit + base) / (total + 1.0)
                    asked[j] = (rate[x_all[:, index]] if index is not None
                                else np.full(len(x_all), rate[0]))
                full = _two_part(asked[j], running)
            if k is None:
                out[j] = full[n:]
            else:
                oof[j][fold == k] = full[:n][fold == k]
    return list(zip(oof, out))


# --------------------------------------------------------------------- nets --

_DEVICE = []                      # set once at import by _probe_device()


def _probe_device():
    """The device the nets will run on, or None, decided at import.

    A GPU that imports but cannot run these kernels would drop every net
    silently on the survey rows (each net is wrapped so that a failure falls
    back to the other estimators). So the real net code is run once on a tiny
    made-up problem -- no survey data exists yet -- on the GPU first, then
    on the CPU, and the first device that gives sane output is used.
    """
    try:
        import torch
    except Exception:
        return None
    fold = np.arange(40, dtype=np.int64) % FOLDS
    x = np.stack([np.arange(44) % 3, 3 + np.arange(44) % 2], axis=1)
    y = np.stack([np.arange(40) % 3, np.arange(40) % 2], axis=1)
    config = dict(name="probe", hidden=8, lr=0.004, wd=0.001, dropout=0.0,
                  steps=(1,), seed=0)
    devices = ["cpu"]
    try:
        if torch.cuda.is_available():
            devices.insert(0, "cuda")
    except Exception:
        pass
    for device in devices:
        try:
            deadline = time.monotonic() + 60.0
            results = (_nets(config, x, 5, y, [3, 2], fold, 4, deadline, device),
                       _split_nets(config, x, 5, y, [3, 2], [True, True], fold,
                                   4, deadline, device))
            for result in results:
                for (fit, query), width in zip(result, (3, 2)):
                    if (fit.shape != (40, width) or query.shape != (4, width)
                            or not np.isfinite(fit).all()
                            or not np.allclose(fit.sum(1), 1.0)
                            or not np.allclose(query.sum(1), 1.0)):
                        raise ValueError("probe output")
            return device
        except Exception:
            continue
        finally:
            LAST_RUN.clear()
    return None


def _nets(config, xidx, n_inputs, Y, widths, fold, n_out, deadline,
          device=None):
    """Multi-task softmax nets for every fold plus all rows, in one batch.

    xidx  (n + n_out, G) global one-hot indices, fit rows first
    Y     (n, J) answer codes, -1 where unknown
    Returns per item (oof (n, w), out (n_out, w)) or None if out of time.
    """
    import torch

    torch.manual_seed(SEED + config.get("seed", 0))
    device = device or _DEVICE[0]
    if device is None:
        raise RuntimeError("no device for the nets")
    if device == "cpu":
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    n, J = Y.shape
    models = FOLDS + 1

    # Items grouped by width so each group's softmax is one reshape.
    order = sorted(range(J), key=lambda j: widths[j])
    groups, offset = [], 0
    for w in sorted(set(widths)):
        members = [j for j in order if widths[j] == w]
        groups.append((w, members, offset))
        offset += w * len(members)
    S = offset

    fold_t = torch.as_tensor(fold, device=device)
    train = torch.ones(models, n, dtype=torch.bool, device=device)
    for k in range(FOLDS):
        train[k] = fold_t != k
    Yt = torch.as_tensor(Y, device=device)
    idx = torch.as_tensor(xidx, device=device)
    G = idx.shape[1]
    stacked = (idx[None] + n_inputs * torch.arange(models, device=device)
               [:, None, None]).reshape(-1, G)        # (models * rows, G)

    # Start every head at its training marginal.
    bias = torch.zeros(models, S, device=device)
    for w, members, off in groups:
        y = Yt[:, members]                              # (n, Jg)
        for k in range(models):
            rows = y[train[k]]
            counts = torch.stack([(rows == c).sum(0) for c in range(w)], -1)
            p = (counts.float() + 0.5)
            p = p / p.sum(-1, keepdim=True)
            bias[k, off:off + w * len(members)] = p.log().reshape(-1)

    hidden = config["hidden"]
    if hidden:
        emb = torch.nn.EmbeddingBag(models * n_inputs, hidden, mode="sum")
        torch.nn.init.normal_(emb.weight, std=0.1)
        b1 = torch.nn.Parameter(torch.zeros(models, 1, hidden, device=device))
        w2 = torch.nn.Parameter(0.01 * torch.randn(models, hidden, S,
                                                   device=device))
        emb = emb.to(device)
        params = [emb.weight, b1, w2]
    else:
        emb = torch.nn.EmbeddingBag(models * n_inputs, S, mode="sum")
        torch.nn.init.zeros_(emb.weight)
        emb = emb.to(device)
        params = [emb.weight]
    b2 = torch.nn.Parameter(bias)
    params.append(b2)
    opt = torch.optim.AdamW(params, lr=config["lr"],
                            weight_decay=config["wd"])
    drop = torch.nn.Dropout(config["dropout"])

    def logits(rows, training):
        rows_total = xidx.shape[0]
        sel = stacked.reshape(models, rows_total, G)[:, :rows].reshape(-1, G)
        z = emb(sel).reshape(models, rows, -1)
        if hidden:
            z = torch.nn.functional.gelu(z + b1)
            if training:
                z = drop(z)
            z = torch.bmm(z, w2)
        return z + b2[:, None, :]

    def log_probs(z):
        parts = []
        for w, members, off in groups:
            block = z[..., off:off + w * len(members)]
            parts.append(block.reshape(*block.shape[:-1], len(members), w)
                         .log_softmax(-1).reshape(block.shape))
        return torch.cat(parts, -1)

    def nll(lp):
        total = 0.0
        for w, members, off in groups:
            y = Yt[:, members]                          # (n, Jg)
            block = lp[:, :, off:off + w * len(members)].reshape(
                models, n, len(members), w)
            picked = block.gather(-1, y.clamp(min=0)[None, ..., None]
                                  .expand(models, -1, -1, 1))[..., 0]
            mask = (y >= 0)[None] & train[:, :, None]
            total = total + -(picked * mask).sum((1, 2))
        if "loss_cells" in config:
            per_row = torch.as_tensor(config["loss_cells"], device=device)
            cells = (train * per_row[None]).sum(1).clamp(min=1)
        else:
            cells = ((Yt >= 0)[None] & train[:, :, None]).sum((1, 2)).clamp(min=1)
        return (total / cells).sum()

    # Column -> item, and each item's columns, in the grouped layout.
    item_cols = {}
    for w, members, off in groups:
        for r, j in enumerate(members):
            item_cols[j] = list(range(off + r * w, off + (r + 1) * w))
    best_loss = torch.full((J,), -float("inf"), device=device)
    best_step = torch.zeros(J, dtype=torch.long, device=device)
    best_oof = torch.zeros(n, S, device=device)
    best_out = torch.zeros(xidx.shape[0] - n, S, device=device)
    arange = torch.arange(n, device=device)
    done, seen = False, False

    step = 0
    for checkpoint in config["steps"]:
        while step < checkpoint:
            if time.monotonic() > deadline:
                done = True
                break
            opt.zero_grad(set_to_none=True)
            loss = nll(log_probs(logits(n, True)))
            loss.backward()
            opt.step()
            step += 1
        if done and not seen:
            return None
        with torch.no_grad():
            lp = log_probs(logits(xidx.shape[0], False))
            oof = lp[fold_t, arange].exp()             # (n, S)
            out = lp[FOLDS, n:].exp()                  # (n_out, S)
            improved = torch.zeros(S, dtype=torch.bool, device=device)
            for w, members, off in groups:
                y = Yt[:, members]
                block = oof[:, off:off + w * len(members)].reshape(
                    n, len(members), w)
                picked = block.gather(-1, y.clamp(min=0)[..., None])[..., 0]
                logp = torch.log(FLOOR + (1.0 - w * FLOOR) * picked)
                known = (y >= 0).float()
                value = (logp * known).sum(0) / known.sum(0).clamp(min=1)
                better = value > best_loss[members]
                best_loss[members] = torch.where(better, value,
                                                 best_loss[members])
                best_step[members] = torch.where(
                    better, torch.full_like(best_step[members], step),
                    best_step[members])
                improved[off:off + w * len(members)] = (
                    better[:, None].expand(-1, w).reshape(-1))
            best_oof[:, improved] = oof[:, improved]
            best_out[:, improved] = out[:, improved]
            seen = True
        if done:
            break
    if not seen:
        return None
    best_oof, best_out = best_oof.double().cpu().numpy(), best_out.double().cpu().numpy()
    LAST_RUN.setdefault("steps", {})[config["name"]] = np.bincount(
        np.searchsorted(config["steps"], best_step.cpu().numpy()),
        minlength=len(config["steps"])).tolist()
    return [(best_oof[:, item_cols[j]], best_out[:, item_cols[j]])
            for j in range(J)]


def _split_targets(Y, widths, gated):
    """Expand each gated response into asked and conditional-answer targets.

    A known sentinel contributes only to the asked head; an ordinary answer
    contributes to both. Unknown cells contribute to neither. The summed head
    NLL is exactly the original cell NLL, before the scoring floor.
    """
    columns, head_widths, layout = [], [], []
    for j, width in enumerate(widths):
        y = Y[:, j]
        if gated[j]:
            K = width - 1
            asked = np.where(y < 0, -1, (y < K).astype(np.int64))
            answer = np.where((y >= 0) & (y < K), y, -1)
            layout.append((len(columns), len(columns) + 1))
            columns.extend([asked, answer])
            head_widths.extend([2, K])
        else:
            layout.append((None, len(columns)))
            columns.append(y)
            head_widths.append(width)
    return np.stack(columns, 1), head_widths, layout


def _split_nets(config, xidx, n_inputs, Y, widths, gated, fold, n_out, deadline,
                device=None):
    heads, head_widths, layout = _split_targets(Y, widths, gated)
    config = dict(config, loss_cells=(Y >= 0).sum(1))
    result = _nets(config, xidx, n_inputs, heads, head_widths, fold, n_out,
                   deadline, device)
    if result is None:
        return None
    parts = []
    for asked, answer in layout:
        if asked is None:
            parts.append(result[answer])
        else:
            parts.append(tuple(_two_part(result[asked][k][:, 1], result[answer][k])
                               for k in (0, 1)))
    return parts


def _pair_inputs(x_all, cards, xidx, n_inputs):
    """Pair features chosen without target labels, shared by all folds."""
    eligible = [c for c in range(len(cards)) if np.unique(x_all[:, c]).size > 1]
    top = sorted(eligible, key=lambda c: (cards[c], c))[:PAIR_COLUMNS]
    extra = []
    for i, a in enumerate(top):
        for b in top[i + 1:]:
            extra.append(n_inputs + x_all[:, a] * cards[b] + x_all[:, b])
            n_inputs += cards[a] * cards[b]
    if not extra:
        return xidx, n_inputs
    return np.concatenate([xidx, np.stack(extra, 1)], 1), n_inputs


# ---------------------------------------------------------------- llm prior --

def _llm_components(frame, targets, values, gated, fit_rows, out_rows, chain):
    """Per item (oof, out) from the baked prior, or None where it cannot apply.

    The prior speaks to the options only. For a gated item the sentinel share
    comes from the chain's own estimate for that row (out of fold for the fit
    rows), and the prior splits the rest.
    """
    if not LLM_PRIOR:
        return [None] * len(targets)
    table, column = None, None
    for name, items in LLM_PRIOR.get("items", {}).items():
        if set(targets) <= set(items):
            table, column = items, LLM_PRIOR.get("context", {}).get(name)
            break
    if table is None:
        return [None] * len(targets)
    if column is not None and column in frame.columns:
        context = frame[column].to_numpy(dtype=object)
    else:
        context = np.full(len(frame), "", dtype=object)
    result = []
    for j, item in enumerate(targets):
        try:
            shares = table[item]
            width = len(values[j])
            mean = np.mean([v for v in shares.values()], axis=0)
            if mean.size != width or (gated[j] and chain[j] is None):
                result.append(None)
                continue
            lookup = {k: np.asarray(v, float) for k, v in shares.items()}

            def rows(index):
                return np.stack([lookup.get(c, mean) for c in context[index]])

            fit, out = rows(fit_rows), rows(out_rows)
            if gated[j]:
                s_fit = chain[j][0][:, -1:]
                s_out = chain[j][1][:, -1:]
                fit = np.concatenate([(1.0 - s_fit) * fit, s_fit], axis=1)
                out = np.concatenate([(1.0 - s_out) * out, s_out], axis=1)
            result.append((fit, out))
        except Exception:
            result.append(None)
    return result


# ----------------------------------------------------------------- stacking --

def _temper(p, temperature):
    if temperature == 1.0:
        return p
    q = np.power(np.maximum(p, 1e-300), 1.0 / temperature)
    return q / q.sum(axis=1, keepdims=True)


def _polish(truth, weights):
    """Finish the EM weights with L-BFGS on the same concave objective."""
    try:
        from scipy.optimize import minimize

        def objective(z):
            w = np.exp(z - z.max())
            w /= w.sum()
            mix = truth @ w
            return (-np.log(mix).mean(),
                    w - (truth * (w / mix[:, None])).mean(axis=0))

        start = np.log(np.maximum(weights, 1e-8))
        result = minimize(objective, start, jac=True, method="L-BFGS-B",
                          options={"maxiter": 300, "ftol": 1e-13, "gtol": 1e-9})
        w = np.exp(result.x - result.x.max())
        w /= w.sum()
        if np.isfinite(w).all() and objective(result.x)[0] <= objective(start)[0]:
            return w
    except Exception:
        pass
    return weights


def _stack(oofs, y, width):
    """Simplex weights then a temperature, both by out-of-fold log score."""
    # The floor is linear in p, so the floored mixture is the mixture of the
    # floored components and EM on those maximises the graded score exactly
    # (the log likelihood is concave in the weights).
    truth = _floored(np.stack([o[np.arange(y.size), y] for o in oofs], 1),
                     width)                            # (n, C)
    weights = np.full(truth.shape[1], 1.0 / truth.shape[1])
    for _ in range(EM_STEPS):
        share = truth * weights
        share /= share.sum(axis=1, keepdims=True)
        weights, previous = share.mean(axis=0), weights
        if np.abs(weights - previous).max() < EM_TOL:
            break
    weights = _polish(truth, weights)
    mixed = _mix(oofs, weights, 1.0)
    best = (-np.inf, 1.0)
    for t in TEMPERATURES:
        q = _temper(mixed, t)
        value = _log_score(q[np.arange(y.size), y], width)
        if value > best[0]:
            best = (value, t)
    return weights, best[1], best[0]


def _split_temper(pair, y, known, width, score):
    """For a gated item, temper whether-asked and the answer separately.

    The mixture's sentinel share and its answer shape come from different
    components and need not be off by the same factor. Both temperatures by
    the out-of-fold score; kept only if they beat the single one.
    Returns ((oof, out), score).
    """
    def apply(p, t_asked, t_answer):
        sentinel = np.clip(p[:, -1], 1e-12, 1.0 - 1e-12)
        logit = np.log1p(-sentinel) - np.log(sentinel)
        asked = 1.0 / (1.0 + np.exp(-logit / t_asked))
        answer = p[:, :-1] / np.maximum(p[:, :-1].sum(1, keepdims=True), 1e-300)
        return _two_part(asked, _temper(answer, t_answer))

    oof, out = pair
    rows = np.arange(y.size)
    best = (score, None)
    for t_asked in TEMPERATURES:
        for t_answer in TEMPERATURES:
            q = apply(oof[known], t_asked, t_answer)
            value = _log_score(q[rows, y], width)
            if value > best[0] + 1e-6:
                best = (value, (t_asked, t_answer))
    if best[1] is None:
        return pair, score
    return (apply(oof, *best[1]), apply(out, *best[1])), best[0]


def _mix(parts, weights, temperature):
    return _temper(sum(w * p for w, p in zip(weights, parts)), temperature)


def _marginal_parts(y, width, fold, n_out):
    """The item's marginal: (oof without each row's fold, out on every row)."""
    known = y >= 0
    counts = np.bincount(fold[known] * width + y[known],
                         minlength=FOLDS * width).reshape(FOLDS, width)
    rest = counts.sum(0) - counts + 0.5
    oof = (rest / rest.sum(1, keepdims=True))[fold]
    return oof, np.tile(_marginal(y[known], width), (n_out, 1))


# ------------------------------------------------------------------ predict --

def _budget(schema, hidden_rows, work):
    """Seconds this call may spend, from what is left of the shared budget."""
    split = schema.get("split", {})
    phase = 2 if hidden_rows == split.get("n_test", -1) else 1
    left = BUDGET[phase] - (time.monotonic() - _IMPORTED)
    calls_left = max(CALLS - _calls_done[0], 1)
    if calls_left == 1:
        share = 1.0
    else:
        share = max(work / max(WORK[phase] - _work_done[0], work),
                    0.5 / calls_left)
    return max(left * 0.8 * min(share, 1.0), 5.0)


def _gates(schema, given, targets, values, cards):
    """Per target its gate as (kind, index, opens) or None, and a stacking
    order with every PREDICT parent ahead of its children."""
    where = {name: ("given", g) for g, name in enumerate(given)}
    where.update({name: ("predict", j) for j, name in enumerate(targets)})
    parents = []
    for name in targets:
        gate = schema["items"][name].get("gate")
        if not gate or gate.get("parent") not in where:
            parents.append(None)
            continue
        kind, index = where[gate["parent"]]
        levels = (values[index] if kind == "predict"
                  else list(schema["items"][gate["parent"]]["values"]))
        opens = [levels.index(v) for v in gate.get("observed_if", [])
                 if v in levels]
        parents.append((kind, index, opens))

    def depth(j):
        levels, seen = 0, set()
        while parents[j] is not None and parents[j][0] == "predict" \
                and j not in seen:
            seen.add(j)
            j = parents[j][1]
            levels += 1
        return levels

    order = sorted(range(len(targets)), key=lambda j: (depth(j), j))
    rank = {j: r for r, j in enumerate(order)}
    for j, gate in enumerate(parents):     # a cycle: the child goes ungated
        if gate and gate[0] == "predict" and rank[gate[1]] > rank[j]:
            parents[j] = None
    return parents, order


def _fit_predict(frame, schema, deadline):
    started = time.monotonic()
    LAST_RUN.clear()
    given, targets = _items(schema)
    J = len(targets)
    gated = [bool(schema["items"][t].get("gate")) for t in targets]
    values = [list(schema["items"][t]["values"]) for t in targets]
    widths = [len(values[j]) + gated[j] for j in range(J)]

    blank = frame[targets].isna().to_numpy()            # (rows, J)
    out_rows = np.flatnonzero(blank.any(axis=1))
    fit_rows = np.flatnonzero(~blank.all(axis=1))

    x = np.zeros((len(frame), len(given)), dtype=np.int64)
    cards = []
    for g, name in enumerate(given):
        x[:, g], card = _given_codes(frame[name],
                                     schema["items"][name]["values"])
        cards.append(card)
    Y = np.stack([_target_codes(frame[t].iloc[fit_rows], values[j], gated[j])
                  for j, t in enumerate(targets)], 1) if J else None
    n = fit_rows.size
    fold = np.random.default_rng(SEED).permutation(n) % FOLDS
    x_fit, x_out = x[fit_rows], x[out_rows]
    parents, order = _gates(schema, given, targets, values, cards)

    # ---- chains: always, they are cheap ------------------------------------
    chain, split, answer, profile = ([None] * J for _ in range(4))
    try:
        profile_keys = _profile_keys(np.concatenate([x_fit, x_out]))
    except Exception:
        profile_keys = None
    for j in range(J):
        y = Y[:, j]
        if (y >= 0).sum() < 20 or time.monotonic() > deadline:
            continue
        try:
            oof, out, columns = _chain(x_fit, x_out, y, widths[j], fold)
            chain[j] = (oof, out)
            if profile_keys is not None:
                try:
                    profile[j] = _profile(profile_keys, n, y, widths[j], fold, chain[j])
                except Exception:
                    profile[j] = None
            if gated[j]:
                K = widths[j] - 1
                asked = np.where(y < 0, -1, (y != K).astype(np.int64))
                a_oof, a_out, _ = _chain(x_fit, x_out, asked,
                                                        2, fold)
                answer[j] = _chain(x_fit, x_out, np.where(y < K, y, -1), K,
                                   fold)[:2]
                split[j] = (_two_part(a_oof[:, 1], answer[j][0]),
                            _two_part(a_out[:, 1], answer[j][1]))
        except Exception:
            chain[j] = split[j] = answer[j] = None

    timing = {"chain": time.monotonic() - started}
    official = [None] * J
    if J and time.monotonic() < deadline:
        try:
            official = _official(x_fit, x_out, cards, Y, widths, gated,
                                 parents, order, fold)
        except Exception:
            official = [None] * J
    timing["official"] = time.monotonic() - started

    # ---- nets: when there are rows enough and time left ----------------------
    nets = {}
    if n >= NET_MIN_ROWS and J and _DEVICE[0] is not None:
        offsets = np.concatenate([[0], np.cumsum(cards)[:-1]])
        x_all = np.concatenate([x_fit, x_out])
        inputs = {False: (x_all + offsets[None, :], int(sum(cards)))}
        inputs[True] = _pair_inputs(x_all, cards, *inputs[False])
        for config in NETS:
            remaining = deadline - time.monotonic()
            if remaining < 10:
                break
            try:
                xidx, n_inputs = inputs[bool(config.get("pairs"))]
                net_deadline = time.monotonic() + remaining * 0.4
                if config.get("split"):
                    result = _split_nets(config, xidx, n_inputs, Y, widths,
                                         gated, fold, out_rows.size, net_deadline)
                else:
                    result = _nets(config, xidx, n_inputs, Y, widths, fold,
                                   out_rows.size, net_deadline)
            except Exception:
                result = None
            if result is not None:
                nets[config["name"]] = result
            timing[config["name"]] = time.monotonic() - started

    llm = _llm_components(frame, targets, values, gated, fit_rows, out_rows,
                          chain)

    # ---- stack per item, parents first ---------------------------------------
    stacked, blocks, report = [None] * J, [None] * J, []
    for j in order:
        width = widths[j]
        known = np.flatnonzero(Y[:, j] >= 0)
        parts = []                                    # (name, oof, out)
        for name, result in (("chain", chain[j]), ("split", split[j]),
                             ("profile", profile[j]),
                             ("official", official[j]), ("llm", llm[j])):
            if result is not None:
                parts.append((name,) + tuple(result))
        for name, result in nets.items():
            parts.append((name,) + tuple(result[j]))
        if answer[j] is not None and parents[j] is not None:
            try:
                parts += _gate_parts(j, parents[j], chain, Y, x_fit, x_out,
                                     cards, fold, answer[j], width)
            except Exception:
                pass
        parts.append(("marginal",) + _marginal_parts(Y[:, j], width, fold,
                                                     out_rows.size))
        parts = [(name, oof, out) for name, oof, out in parts
                 if np.isfinite(oof).all() and np.isfinite(out).all()]
        try:
            weights, temperature, score = _stack(
                [oof[known] for _, oof, _ in parts], Y[known, j], width)
            stacked[j] = (_mix([oof for _, oof, _ in parts], weights,
                               temperature),
                          _mix([out for _, _, out in parts], weights,
                               temperature))
            if gated[j]:
                stacked[j], score = _split_temper(stacked[j], Y[known, j],
                                                  known, width, score)
            report.append((targets[j],
                           dict(zip([p[0] for p in parts], weights.round(2))),
                           temperature, score))
        except Exception:
            stacked[j] = None
        if stacked[j] is None or not np.isfinite(stacked[j][1]).all():
            stacked[j] = _marginal_parts(Y[:, j], width, fold, out_rows.size)
        blocks[j] = stacked[j][1][blank[out_rows, j]]

    steps = LAST_RUN.get("steps", {})
    LAST_RUN.clear()
    LAST_RUN["steps"] = steps
    timing["stack"] = time.monotonic() - started
    LAST_RUN.update({"report": report, "timing": timing,
                     "nets": sorted(nets)})
    return _assemble(blank, blocks)


def _gate_parts(j, gate, chain, Y, x_fit, x_out, cards, fold, answer, width):
    """A gate component without reusing label-selected parent stack weights."""
    kind, index, _ = gate
    if kind == "predict":
        if chain[index] is None:
            return []
        parent_oof, parent_out = chain[index]
        parent_seen = Y[:, index]
    else:
        eye = np.eye(cards[index])
        parent_oof, parent_out = eye[x_fit[:, index]], eye[x_out[:, index]]
        parent_seen = x_fit[:, index]
    y = Y[:, j]
    asked = np.where(y < 0, -1, (y != width - 1).astype(np.int64))
    a_oof, a_out = _gate_asked(parent_oof, parent_out, parent_seen, asked, fold)
    return [("gate", _two_part(a_oof, answer[0]), _two_part(a_out, answer[1]))]


def _assemble(blank, blocks):
    """One plain list per blank cell: rows top to bottom, items in schema order."""
    lists, pointer = [], [0] * len(blocks)
    for block in blocks:
        block = np.where(np.isfinite(block), np.maximum(block, 0.0), 0.0)
        sums = block.sum(axis=1, keepdims=True)
        block = np.where(sums > 0, block / np.where(sums > 0, sums, 1.0),
                         1.0 / max(block.shape[1], 1))
        lists.append(block.tolist())
    out = []
    for row in np.flatnonzero(blank.any(axis=1)):
        for j in np.flatnonzero(blank[row]):
            out.append(lists[j][pointer[j]])
            pointer[j] += 1
    return out


# ----------------------------------------------------- pretrained tabular models --

_FM = {}


def _fm_model(name, n_given):
    """The classifier for one pretrained model, built once, or None."""
    key = (name, n_given)
    if key not in _FM:
        model = None
        try:
            library, filename, members = FM_SPECS[name]
            path = os.path.join(HERE, filename)
            if _DEVICE[0] == "cuda" and os.path.exists(path):
                if library == "tabpfn":
                    from tabpfn import TabPFNClassifier
                    model = TabPFNClassifier(
                        model_path=path, device="cuda", n_estimators=members,
                        categorical_features_indices=list(range(n_given)),
                        random_state=0, ignore_pretraining_limits=True)
                else:
                    from tabicl import TabICLClassifier

                    class _OneLoad(TabICLClassifier):
                        # fit() would re-read the checkpoint every time
                        def _load_model(self):
                            if not hasattr(self, "model_"):
                                super()._load_model()

                    model = _OneLoad(n_estimators=members, model_path=path,
                                     allow_auto_download=False, device="cuda",
                                     random_state=0)
        except Exception:
            model = None
        _FM[key] = model
    return _FM[key]


def _fm_parts(name, frame, schema, given, targets, fit, query, deadline):
    """Per item j, one model's probabilities (query rows, width) fitted on `fit`.

    Each item is its own in-context problem on the GIVEN columns. Items are
    done in schema order until the deadline; the rest are absent.
    """
    model = _fm_model(name, len(given))
    parts = {}
    if model is None or fit.size < 50 or query.size == 0:
        return parts
    x = np.stack([_given_codes(frame[g], schema["items"][g]["values"])[0]
                  for g in given], 1).astype(np.float32)
    for j, item in enumerate(targets):
        if time.monotonic() > deadline:
            break
        record = schema["items"][item]
        gated = bool(record.get("gate"))
        width = len(record["values"]) + gated
        y = _target_codes(frame[item].iloc[fit], list(record["values"]), gated)
        keep = y >= 0
        if np.unique(y[keep]).size < 2:
            continue
        try:
            model.fit(x[fit[keep]], y[keep])
            q = model.predict_proba(x[query])
            p = np.zeros((query.size, width))
            p[:, np.asarray(model.classes_, dtype=np.int64)] = q
            if np.isfinite(p).all() and (p.sum(1) > 0).all():
                parts[j] = p / p.sum(1, keepdims=True)
        except Exception:
            continue
    return parts


def _fm_combos():
    """Every weight vector over the models on the FM_W grid that sums to <= 1."""
    combos = [()]
    for _ in FM_NAMES:
        combos = [c + (w,) for c in combos for w in FM_W
                  if sum(c) + w <= 1.0 + 1e-9]
    return combos


def _fm_weights(blocks):
    """Mixing weights per item, shrunk to the instrument-wide choice.

    blocks: per item (a, [b_f or None], width), the held-out probabilities of
    the true answer under the stack (a) and under each model (b_f). The pool
    is linear, so the true-answer probability of a mixture is the mixture of
    these and every weight vector can be scored exactly. An item picks the
    vector that maximises its own held-out log score plus FM_SHRINK cells of
    the instrument-wide mean curve, so an item with little evidence of its own
    follows the instrument. Returns (instrument weights, {item: weights}).
    """
    combos = _fm_combos()
    zero = combos.index(tuple(0.0 for _ in FM_NAMES))
    keys = list(blocks)
    S = np.zeros((len(keys), len(combos)))
    cells = 0
    for r, j in enumerate(keys):
        a, bs, width = blocks[j]
        cells += a.size
        for c, weights in enumerate(combos):
            p = (1.0 - sum(weights)) * a
            for w, b in zip(weights, bs):
                p = p + w * (a if b is None else b)
            S[r, c] = np.log(_floored(p, width)).sum()
    total = S.sum(0)
    best = int(total.argmax())
    if cells == 0 or (total[best] - total[zero]) / max(cells, 1) < RECAL_MIN_GAIN:
        return combos[zero], {}
    curve = total / cells
    chosen = {}
    for r, j in enumerate(keys):
        if all(b is None for b in blocks[j][1]):
            continue
        chosen[j] = combos[int((S[r] + FM_SHRINK * curve).argmax())]
    return combos[best], chosen


def _fm_mix(p, parts_j, weights):
    """Stack vector(s) p mixed with each model's vectors; a missing model's
    share stays with the stack."""
    out = (1.0 - sum(weights)) * p
    for w, t in zip(weights, parts_j):
        out = out + w * (p if t is None else t)
    return out


# ------------------------------------------------------------ recalibration --

def _item_blocks(blank, targets):
    """Per item, the positions of its cells in the canonical cell order."""
    order, position = {j: [] for j in range(len(targets))}, 0
    for row in np.flatnonzero(blank.any(axis=1)):
        for j in np.flatnonzero(blank[row]):
            order[j].append(position)
            position += 1
    return order


def _item_rows(blank, j):
    """Rank, among the blank rows, of each row where item j is blank."""
    rows = np.flatnonzero(blank.any(axis=1))
    return np.flatnonzero(blank[rows, j])


def _smoothed(frame_rows, item, values, gated):
    codes = _target_codes(frame_rows[item], values, gated)
    width = len(values) + gated
    return _marginal(codes[codes >= 0], width)


def _recal_apply(p, marginal, temperature, mix):
    return (1.0 - mix) * _temper(p, temperature) + mix * marginal[None, :]


def _recalibrate(frame, schema, given, targets, blank, deadline):
    """Model weights, a temperature and a marginal mix from held-out rows.

    The entire procedure, and each pretrained model beside it, is rerun on 80%
    of the visible rows and scored on the other 20%, which no part of that run
    saw. Returns (instrument weights, {item: weights}, temperature, mix).
    """
    none = tuple(0.0 for _ in FM_NAMES)
    visible = np.flatnonzero(~blank.any(axis=1))
    if visible.size < 500:
        return none, {}, 1.0, 0.0
    rng = np.random.default_rng(SEED + 7)
    held = np.zeros(visible.size, bool)
    held[rng.permutation(visible.size)[:int(visible.size * RECAL_SHARE)]] = True
    inner = frame.iloc[visible].reset_index(drop=True).copy()
    truth = inner.loc[held, targets].copy()
    for item in targets:
        inner[item] = inner[item].astype(object)
        inner.loc[held, item] = np.nan
    now = time.monotonic()
    vectors = _fit_predict(inner, schema, now + (deadline - now) * 0.45)
    fms = []
    for f, name in enumerate(FM_NAMES):
        now = time.monotonic()
        share = (deadline - now) / (len(FM_NAMES) - f)
        fms.append(_fm_parts(name, inner, schema, given, targets,
                             np.flatnonzero(~held), np.flatnonzero(held),
                             now + share))
    inner_blank = inner[targets].isna().to_numpy()
    cells = _item_blocks(inner_blank, targets)
    gated = [bool(schema["items"][t].get("gate")) for t in targets]
    values = [list(schema["items"][t]["values"]) for t in targets]
    fit = inner.loc[~held]
    full, scored = {}, {}
    for j, item in enumerate(targets):
        width = len(values[j]) + gated[j]
        y = _target_codes(truth[item], values[j], gated[j])
        keep = np.flatnonzero(y >= 0)
        if keep.size == 0:
            continue
        p = np.asarray([vectors[c] for c in cells[j]], float)[keep]
        rows = _item_rows(inner_blank, j)[keep]
        ts = [fm[j][rows] if j in fm else None for fm in fms]
        yk = y[keep]
        pick = np.arange(yk.size)
        full[j] = (p, ts, yk, _smoothed(fit, item, values[j], gated[j]), width)
        scored[j] = (p[pick, yk], [None if t is None else t[pick, yk] for t in ts],
                     width)
    if not full:
        return none, {}, 1.0, 0.0
    weights, per_item = _fm_weights(scored)
    cells_total = sum(b[2].size for b in full.values())

    def value(temperature, mix):
        total = 0.0
        for j, (p, ts, y, marginal, width) in full.items():
            q = _fm_mix(p, ts, per_item.get(j, weights))
            q = _recal_apply(q, marginal, temperature, mix)
            total += np.log(_floored(q[np.arange(y.size), y], width)).sum()
        return total / cells_total

    plain = value(1.0, 0.0)
    calibrated = (plain, 1.0, 0.0)
    for temperature in RECAL_T:
        for mix in RECAL_MIX:
            v = value(temperature, mix)
            if v > calibrated[0]:
                calibrated = (v, temperature, mix)
    if calibrated[0] - plain < RECAL_MIN_GAIN:
        return weights, per_item, 1.0, 0.0
    return weights, per_item, calibrated[1], calibrated[2]


def _unfloor(vector):
    """Report the vector whose floored version is our belief.

    The grader scores log(FLOOR + (1 - K FLOOR) p). For a belief pi the
    expected score is maximised by p with FLOOR + (1 - K FLOOR) p = pi, i.e.
    p proportional to max(pi - FLOOR, 0): mass we do not hold with at least
    the floor's confidence is the floor's to give, not ours.
    """
    p = np.asarray(vector, dtype=float)
    q = np.maximum(p - FLOOR, 0.0)
    total = q.sum()
    if not np.isfinite(total) or total <= 0:
        return vector
    return (q / total).tolist()


def predict(frame, schema):
    vectors = _predict(frame, schema)
    try:
        return [_unfloor(v) for v in vectors]
    except Exception:
        return vectors


def _predict(frame, schema):
    started = time.monotonic()
    given, targets = _items(schema)
    blank = frame[targets].isna().to_numpy()
    out_rows = np.flatnonzero(blank.any(axis=1))
    budget = _budget(schema, out_rows.size, len(targets) * len(frame))
    end = started + budget
    none = tuple(0.0 for _ in FM_NAMES)
    try:
        weights, per_item, temperature, mix = _recalibrate(
            frame, schema, given, targets, blank, started + budget * 0.45)
    except Exception:
        weights, per_item, temperature, mix = none, {}, 1.0, 0.0
    used = [f for f in range(len(FM_NAMES))
            if weights[f] > 0 or any(w[f] > 0 for w in per_item.values())]
    now = time.monotonic()
    stack_end = now + (end - now) * 0.5 if used else end
    vectors = _fit_predict(frame, schema, stack_end)
    LAST_RUN["recalibration"] = (weights, temperature, mix)
    LAST_RUN["fm_item_weights"] = len(per_item)
    _calls_done[0] += 1
    _work_done[0] += len(targets) * len(frame)
    if not used and (temperature, mix) == (1.0, 0.0):
        return vectors
    try:
        visible_rows = np.flatnonzero(~blank.any(axis=1))
        fms = [{} for _ in FM_NAMES]
        for position, f in enumerate(used):
            now = time.monotonic()
            share = (end - 2.0 - now) / (len(used) - position)
            fms[f] = _fm_parts(FM_NAMES[f], frame, schema, given, targets,
                               visible_rows, out_rows, now + share)
        LAST_RUN["fm_items"] = [len(fm) for fm in fms]
        visible = frame.iloc[visible_rows]
        cells = _item_blocks(blank, targets)
        out = list(vectors)
        for j, item in enumerate(targets):
            if not cells[j]:
                continue
            record = schema["items"][item]
            marginal = _smoothed(visible, item, list(record["values"]),
                                 bool(record.get("gate")))
            p = np.asarray([vectors[c] for c in cells[j]], float)
            rows = _item_rows(blank, j)
            ts = [fm[j][rows] if j in fm else None for fm in fms]
            q = _fm_mix(p, ts, per_item.get(j, weights))
            q = _recal_apply(q, marginal, temperature, mix)
            q = q / q.sum(axis=1, keepdims=True)
            if not np.isfinite(q).all():
                continue
            for c, row in zip(cells[j], q.tolist()):
                out[c] = row
        return out
    except Exception:
        return vectors


def _guard():
    """Fail at import, before any survey frame exists, if the bundled model
    cannot run here. The final phase scores one successful run, and a run
    without the model would silently be a different, weaker method."""
    try:
        if _DEVICE[0] != "cuda":
            raise RuntimeError
        x = (np.arange(240).reshape(80, 3) % 5).astype(np.float32)
        y = (np.arange(80) % 2).astype(np.int64)
        for name in FM_NAMES:
            model = _fm_model(name, 3)
            model.fit(x[:60], y[:60])
            p = model.predict_proba(x[60:])
            if p.shape != (20, 2) or not np.isfinite(p).all():
                raise RuntimeError
    except Exception:
        raise RuntimeError("foreist: bundled model did not load") from None
    finally:
        _FM.clear()


_DEVICE.append(_probe_device())
_guard()
