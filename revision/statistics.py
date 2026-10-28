"""Post-hoc ranking and risk-control procedures; no training or inference.

All guarantees below are conditional on independent within-class calibration
observations and a fixed model/score. They do not validate those assumptions.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom

from poultryrisk.risk import transform, cp_upper


SCORES = ('msp', 'msp_temperature', 'margin', 'negative_entropy', 'doctor', 'energy')


def confidence_scores(p, logits, temperature=1.0):
    """Larger means more confident. DOCTOR uses the negative D_alpha odds.

    D_alpha = g/(1-g), g=1-sum(p**2). The sign reversal supplies the
    acceptance convention s>=lambda and is monotone in sum(p**2).
    No input perturbation or training-dependent DOCTOR variant is used.
    """
    p = np.asarray(p, dtype=float)
    ordered = np.sort(p, axis=1)
    g = 1.0 - np.square(p).sum(axis=1)
    return {
        'msp': p.max(axis=1),
        'msp_temperature': transform(p, temperature).max(axis=1),
        'margin': ordered[:, -1]-ordered[:, -2],
        'negative_entropy': (p*np.log(np.clip(p, 1e-300, 1))).sum(axis=1),
        'doctor': -g/np.clip(1-g, 1e-300, None),
        'energy': logsumexp(np.asarray(logits, dtype=float), axis=1),
    }


def aurc(errors, scores):
    """Discrete mean risk at ranks 1..n with exact random-tie expectation.

    For a tie block of m observations and e errors, the expected number of
    errors in its first j entries is j*e/m. E-AURC subtracts the discrete
    oracle on the identical examples. This avoids dependence on saved ID order.
    """
    errors = np.asarray(errors, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if errors.ndim != 1 or scores.shape != errors.shape or not len(errors):
        raise ValueError('Nonempty aligned one-dimensional errors and scores required')
    if np.any((errors != 0) & (errors != 1)) or not np.all(np.isfinite(scores)):
        raise ValueError('Errors must be binary and scores finite')
    order = np.argsort(-scores, kind='stable')
    ranked_s, ranked_e = scores[order], errors[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ranked_s) != 0)+1]
    stops = np.r_[starts[1:], len(errors)]
    risk_sum, previous_errors = 0.0, 0
    for start, stop in zip(starts, stops):
        m, e = stop-start, int(ranked_e[start:stop].sum())
        j = np.arange(1, m+1)
        risk_sum += float(((previous_errors+j*e/m)/(start+j)).sum())
        previous_errors += e
    n, k = len(errors), int(errors.sum())
    oracle = float((np.arange(1, k+1)/np.arange(n-k+1, n+1)).sum()/n) if k else 0.0
    value = risk_sum/n
    return {'aurc': float(value), 'oracle_aurc': oracle, 'e_aurc': float(value-oracle)}


def risk_coverage_curve(y, pred, scores):
    """All attainable >= threshold points, including the all-defer endpoint."""
    y, pred, scores = np.asarray(y), np.asarray(pred), np.asarray(scores)
    order = np.argsort(-scores, kind='stable')
    s, yy, wrong = scores[order], y[order], (pred != y)[order]
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(y)-1]
    counts = np.array([(y == c).sum() for c in range(4)])
    errors = np.column_stack([np.cumsum(wrong & (yy == c)) for c in range(4)])
    accepted = np.column_stack([np.cumsum(yy == c) for c in range(4)])
    rows = [{'threshold': None, 'accepted': 0, 'coverage': 0.0,
             'selective_error': None, 'worst_class_automatic_error': 0.0,
             **{f'automatic_error_{c}': 0.0 for c in range(4)},
             **{f'coverage_{c}': 0.0 for c in range(4)}}]
    for i in ends:
        risks = errors[i]/counts
        rows.append({'threshold': float(s[i]), 'accepted': int(i+1),
                     'coverage': float((i+1)/len(y)),
                     'selective_error': float(errors[i].sum()/(i+1)),
                     'worst_class_automatic_error': float(risks.max()),
                     **{f'automatic_error_{c}': float(risks[c]) for c in range(4)},
                     **{f'coverage_{c}': float(accepted[i, c]/counts[c]) for c in range(4)}})
    return rows


def validation_grid(scores, size=50):
    """50 validation quantiles, deduplicated and ordered strict to permissive."""
    return np.unique(np.quantile(scores, np.linspace(0, 1, size)))[::-1]


def class_error_counts(y, pred, accept):
    return np.array([((y == c) & (pred != y) & accept).sum() for c in range(4)], dtype=int)


def selected_score_threshold(y, pred, scores, grid, alpha=0.1, safety=0.5):
    """Validation selection only, max coverage, first grid entry wins ties."""
    n = np.array([(y == c).sum() for c in range(4)])
    best, coverage = np.inf, -1.0
    for threshold in grid:
        accept = scores >= threshold
        risks = class_error_counts(y, pred, accept)/n
        if np.all(risks <= alpha*safety) and accept.mean() > coverage:
            best, coverage = float(threshold), float(accept.mean())
    return best


def cp_decision(y, pred, scores, threshold, class_delta, alpha=0.1):
    accept = scores >= threshold
    n = np.array([(y == c).sum() for c in range(4)])
    k = class_error_counts(y, pred, accept)
    bounds = np.array([cp_upper(int(kk), int(nn), class_delta) for kk, nn in zip(k, n)])
    return bool(np.all(bounds <= alpha)), n, k, bounds


def fixed_sequence_from_counts(counts, ns, alpha=0.1, delta=0.05):
    """Return index of last consecutively rejected IUT null, or -1.

    counts has shape (threshold, class), ordered strict to permissive. The
    null is that at least one class has risk greater than alpha. No threshold
    multiplicity factor is needed for a fixed sequence stopped at first failure.
    """
    counts, ns = np.asarray(counts), np.asarray(ns)
    pvalues = binom.cdf(counts, ns[None, :], alpha).max(axis=1)
    first = np.flatnonzero(pvalues > delta)
    last = int(first[0]-1) if len(first) else len(pvalues)-1
    return last, pvalues


def ltt_threshold(y, pred, scores, grid, alpha=0.1, delta=0.05):
    n = np.array([(y == c).sum() for c in range(4)])
    counts = np.array([class_error_counts(y, pred, scores >= threshold) for threshold in grid])
    last, pvalues = fixed_sequence_from_counts(counts, n, alpha, delta)
    return (float(grid[last]) if last >= 0 else np.inf), last, pvalues


def crc_threshold(y, pred, scores, alpha=0.1):
    """Exact discrete classwise CRC with right-limit thresholds.

    Loss is 1{score>=threshold AND incorrect}. Use nextafter at error scores
    to exclude every tied error at the boundary; lambda=infinity is always
    structural all-defer. The risk for max_c lambda_c is bounded in expectation
    for each class by monotonicity. This is not a high-probability certificate.
    """
    y, pred, scores = np.asarray(y), np.asarray(pred), np.asarray(scores, dtype=float)
    thresholds = []
    for c in range(4):
        n = int((y == c).sum())
        if n == 0 or 1/(n+1) > alpha:
            thresholds.append(np.inf)
            continue
        errors = np.sort(scores[(y == c) & (pred != y)])[::-1]
        allowed = int(np.floor(np.nextafter(alpha*(n+1)-1, np.inf)))
        if len(errors) <= allowed:
            thresholds.append(-np.inf)
        else:
            thresholds.append(float(np.nextafter(errors[allowed], np.inf)))
    return float(max(thresholds)), thresholds


def k_max(n, class_delta, alpha=0.1):
    """Largest error count whose one-sided CP bound meets the budget."""
    valid = np.flatnonzero(binom.cdf(np.arange(n+1), n, alpha) <= class_delta)
    return int(valid[-1]) if len(valid) else -1
