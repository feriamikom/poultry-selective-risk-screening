"""Exploratory C1-C5 analyses of saved predictions, never primary re-analysis."""
import copy
import json

import numpy as np
from scipy.stats import binom

from poultryrisk.risk import apply_policy, cp_upper, tune_policy
from .common import CLASSES
from .statistics import (SCORES, aurc, confidence_scores, risk_coverage_curve,
                         validation_grid, selected_score_threshold, cp_decision,
                         ltt_threshold, crc_threshold, class_error_counts, k_max)


LABEL = 'POST_HOC_EXPLORATORY; primary 7/24 unchanged'
EXPLORATORY_FAMILY = 12*6*2  # model/seed/precision x scores x CP or LTT


def _number(value):
    return float(value) if np.isfinite(value) else None


def _metrics(y, pred, accept):
    accept = np.asarray(accept, dtype=bool)
    error = accept & (pred != y)
    rows = {'n_test': len(y), 'accepted': int(accept.sum()),
            'errors': int(error.sum()), 'coverage': float(accept.mean()),
            'selective_error': float(error.sum()/accept.sum()) if accept.any() else None}
    risks = []
    for c, name in enumerate(CLASSES):
        mask = y == c
        n, a, k = int(mask.sum()), int((mask & accept).sum()), int((mask & error).sum())
        rows.update({f'n_{name}': n, f'accepted_{name}': a, f'errors_{name}': k,
                     f'coverage_{name}': a/n if n else None,
                     f'automatic_error_{name}': k/n if n else None})
        risks.append(k/n if n else np.nan)
    rows['worst_class_automatic_error'] = float(np.nanmax(risks))
    rows['newcastle_false_healthy'] = int(((y == 3) & (pred == 0) & accept).sum())
    rows['newcastle_false_healthy_rate'] = rows['newcastle_false_healthy']/rows['n_newcastle']
    return rows


def _base(backbone, seed, fmt, **extra):
    return {'analysis': LABEL, 'backbone': backbone, 'seed': seed, 'format': fmt, **extra}


def comparators(ctx):
    rankings, curves, procedures, selections = [], [], [], []
    for backbone, seed, path in ctx.runs():
        selected = ctx.read_json(path/'selected_policies.json')
        existing = ctx.read_json(path/'policies.json')
        for fmt in ('fp32', 'int8'):
            data = {s: ctx.predictions(path, fmt, s) for s in ('val', 'calibration', 'test')}
            prediction = {s: d['p'].argmax(axis=1) for s, d in data.items()}
            scores = {s: confidence_scores(d['p'], d['logits'], selected[fmt]['temperature'])
                      for s, d in data.items()}
            cal, test = data['calibration'], data['test']
            for score_name in SCORES:
                base = _base(backbone, seed, fmt, score=score_name)
                sv, sc, st = (scores[s][score_name] for s in ('val', 'calibration', 'test'))
                rankings.append({**base, 'tie_handling': 'exact expected risk over uniformly permuted tied scores',
                                 'n_test': len(test['y']),
                                 **aurc(prediction['test'] != test['y'], st)})
                for row in risk_coverage_curve(test['y'], prediction['test'], st):
                    renamed = {}
                    for key, value in row.items():
                        for c, name in enumerate(CLASSES):
                            if key == f'automatic_error_{c}': key = f'automatic_error_{name}'
                            if key == f'coverage_{c}': key = f'coverage_{name}'
                        renamed[key] = value
                    curves.append({**base, **renamed})
                grid = validation_grid(sv, 50)
                # Match the original global search's permissive-to-strict grid
                # order and first-candidate tie handling; LTT uses the reverse.
                threshold = selected_score_threshold(data['val']['y'], prediction['val'], sv, grid[::-1])
                for accounting, procedure_delta in [('per_procedure', .05),
                                                    ('bonferroni_144_procedures', .05/EXPLORATORY_FAMILY)]:
                    passed, ns, ks, bounds = cp_decision(cal['y'], prediction['calibration'], sc,
                                                       threshold, procedure_delta/4)
                    structural = not np.isfinite(threshold)
                    accepted = (st >= threshold) if passed and not structural else np.zeros(len(st), bool)
                    status = 'TRIVIAL_ALL_DEFER' if structural else ('POSTHOC_CP_PASS' if passed else 'POSTHOC_CP_FAIL_ALL_DEFER')
                    procedures.append({**base, 'procedure': 'validation_selection_then_CP',
                                       'accounting': accounting, 'procedure_delta': procedure_delta,
                                       'class_delta': procedure_delta/4, 'exploratory_family_size': EXPLORATORY_FAMILY,
                                       'threshold': _number(threshold), 'all_defer_threshold': structural,
                                       'guarantee_type': 'conditional_theoretical_high_probability_classwise; CP_Bonferroni; posthoc_not_prospective',
                                       'status': status,
                                       **{f'cal_UB_{name}': float(bounds[c]) for c, name in enumerate(CLASSES)},
                                       **_metrics(test['y'], prediction['test'], accepted)})
                    ltt_t, last, pvalues = ltt_threshold(cal['y'], prediction['calibration'], sc,
                                                       grid, delta=procedure_delta)
                    accepted = st >= ltt_t
                    procedures.append({**base, 'procedure': 'LTT_fixed_sequence_IUT',
                                       'accounting': accounting, 'procedure_delta': procedure_delta,
                                       'class_delta': procedure_delta, 'exploratory_family_size': EXPLORATORY_FAMILY,
                                       'threshold': _number(ltt_t), 'all_defer_threshold': last < 0,
                                       'selected_grid_index': last, 'grid_size': len(grid),
                                       'selected_p_IUT': float(pvalues[last]) if last >= 0 else None,
                                       'guarantee_type': 'conditional_theoretical_high_probability_classwise; LTT_fixed_sequence_IUT; posthoc_not_prospective',
                                       'status': 'POSTHOC_LTT_PASS' if last >= 0 else 'POSTHOC_LTT_NO_REJECTION_ALL_DEFER',
                                       **_metrics(test['y'], prediction['test'], accepted)})
                    selections.append({**base, 'accounting': accounting,
                                       'validation_CP_threshold': _number(threshold),
                                       'validation_quantile_grid': json.dumps(grid.tolist()),
                                       'LTT_calibration_threshold': _number(ltt_t),
                                       'LTT_pvalues_strict_to_permissive': json.dumps(pvalues.tolist())})
                crc_t, crc_ts = crc_threshold(cal['y'], prediction['calibration'], sc)
                procedures.append({**base, 'procedure': 'classwise_conformal_risk_control',
                                   'accounting': 'expectation_no_delta', 'procedure_delta': None,
                                   'class_delta': None, 'exploratory_family_size': None,
                                   'threshold': _number(crc_t), 'all_defer_threshold': crc_t == np.inf,
                                   'guarantee_type': 'conditional_theoretical_expected_classwise_risk; exchangeability; NOT_high_probability; posthoc',
                                   'status': 'POSTHOC_CRC_EXPECTATION_RULE' if crc_t != np.inf else 'TRIVIAL_ALL_DEFER',
                                   **{f'CRC_threshold_{name}': _number(crc_ts[c]) for c, name in enumerate(CLASSES)},
                                   **_metrics(test['y'], prediction['test'], st >= crc_t)})
            for name in ('maxsoftmax_090', 'temperature_validation_rule', 'class_conditional_conformal'):
                pred, accepted = apply_policy(test['p'], existing[fmt][name])
                procedures.append({**_base(backbone, seed, fmt, score='stored_original_policy'),
                                   'procedure': name, 'accounting': 'original_baseline',
                                   'procedure_delta': None, 'class_delta': None,
                                   'exploratory_family_size': None,
                                   'guarantee_type': ('class_conditional_marginal_set_coverage; NOT_high_probability'
                                                      if name == 'class_conditional_conformal' else 'empirical_only'),
                                   'status': 'EXISTING_BASELINE_REPORTED_POSTHOC',
                                   **_metrics(test['y'], pred, accepted)})
            print(f'C1-C3 {backbone} seed={seed} {fmt}: scores/procedures complete', flush=True)
    ctx.table('C2_score_ranking_AURC', rankings)
    ctx.table('C2_risk_coverage_curves', curves)
    ctx.table('C3_risk_control_comparators', procedures)
    ctx.table('C3_validation_grids_and_LTT_pvalues', selections)


def _variants(cfg, rule):
    """Each variation retains all 24 original slots, even when slots coincide."""
    yield 'multiplicity', 'primary_96_bounds', cfg, rule == 'proposed', None, .05/96
    yield 'multiplicity', 'IUT_24_rules', cfg, rule == 'proposed', None, .05/24
    yield 'multiplicity', 'within_run_4_rules_16_bounds', cfg, rule == 'proposed', None, .05/16
    yield 'multiplicity', 'unadjusted_per_class', cfg, rule == 'proposed', None, .05
    for fraction in (.25, .5, .75, 1.0):
        changed = copy.deepcopy(cfg); changed['tuning_safety_fraction'] = fraction
        yield 'safety_fraction', str(fraction), changed, rule == 'proposed', None, .05/96
    for mode, temperature in [('temperature_on', None), ('temperature_off_T1', 1.)]:
        yield 'temperature', mode, cfg, rule == 'proposed', temperature, .05/96
    for mode in ('global', 'class_quantile', 'class_quantile_margin', 'global_fine'):
        changed = copy.deepcopy(cfg)
        if mode == 'class_quantile': changed['margin_grid'] = [0.]
        if mode == 'global_fine': changed['threshold_grid'] = np.round(np.arange(.5, 1., .005), 3).tolist()
        yield 'threshold_family', mode, changed, mode.startswith('class_'), None, .05/96
    yield 'search_budget', 'primary_grid', cfg, rule == 'proposed', None, .05/96
    changed = copy.deepcopy(cfg)
    changed['quantile_grid'] = np.linspace(0, 1, 21).tolist()
    changed['margin_grid'] = np.round(np.arange(0, .301, .025), 3).tolist()
    yield 'search_budget', 'fine_q21_margin13', changed, rule == 'proposed', None, .05/96


def ablations(ctx):
    details, subsamples, mechanism = [], [], []
    cfg = ctx.cfg
    # Common subsamples across policies/seeds/precisions, drawn within each class.
    # Do not resample model seeds or treat repeated images as independent data.
    first_run = next(iter(ctx.runs()))[2]
    reference = ctx.predictions(first_run, 'fp32', 'calibration')
    ycal = reference['y']
    rng = np.random.default_rng(20261027)
    sample_indices = {}
    for fraction in (.25, .5, .75):
        cls = [np.flatnonzero(ycal == c) for c in range(4)]
        sizes = [max(1, int(np.floor(len(ind)*fraction))) for ind in cls]
        sample_indices[fraction] = ([np.array([rng.choice(ind, n, replace=False) for _ in range(500)])
                                     for ind, n in zip(cls, sizes)], sizes)
    for backbone, seed, path in ctx.runs():
        selected = ctx.read_json(path/'selected_policies.json')
        certificates = ctx.read_json(path/'certificates.json')
        for fmt in ('fp32', 'int8'):
            val, cal, test = (ctx.predictions(path, fmt, s) for s in ('val', 'calibration', 'test'))
            if not np.array_equal(cal['image_id'], reference['image_id']):
                raise ValueError('Calibration identity mismatch across ablation runs')
            for rule in ('global', 'proposed'):
                base = _base(backbone, seed, fmt, original_rule_slot=rule,
                             original_primary_status=certificates[fmt][rule]['status'])
                original = selected[fmt][rule]
                for factor, variant, new_cfg, classwise, temp, d in _variants(cfg, rule):
                    temperature = original['temperature'] if temp is None else temp
                    if factor == 'multiplicity':
                        policy = original
                    else:
                        policy = tune_policy(val['p'], val['y'], new_cfg,
                                             classwise=classwise, temperature=temperature)
                    pred_cal, acc_cal = apply_policy(cal['p'], policy)
                    k = class_error_counts(cal['y'], pred_cal, acc_cal)
                    ns = np.array([(cal['y'] == c).sum() for c in range(4)])
                    bounds = np.array([cp_upper(int(kk), int(nn), d) for kk, nn in zip(k, ns)])
                    structural = policy.get('all_defer', False) or min(policy['thresholds']) > 1
                    passed = bool(np.all(bounds <= .1)) and not structural
                    pred, accepted = apply_policy(test['p'], policy)
                    as_selected = _metrics(test['y'], pred, accepted)
                    deployed = _metrics(test['y'], pred, accepted if passed else np.zeros(len(accepted), bool))
                    details.append({**base, 'factor': factor, 'variant': variant,
                                    'class_delta': d, 'posthoc_passes': passed,
                                    'structural_all_defer': bool(structural),
                                    'temperature': temperature,
                                    'thresholds_healthy_coccidiosis_salmonella_newcastle': json.dumps(policy['thresholds']),
                                    'margin': policy['margin'],
                                    'slot_note': ('both_original_slots_may_coincide_for_family_ablation'
                                                  if factor == 'threshold_family' else 'original_24_slot_family'),
                                    **{f'cal_errors_{name}': int(k[c]) for c, name in enumerate(CLASSES)},
                                    **{f'cal_UB_{name}': float(bounds[c]) for c, name in enumerate(CLASSES)},
                                    'as_selected_test_coverage': as_selected['coverage'],
                                    'as_selected_test_worst_class_error': as_selected['worst_class_automatic_error'],
                                    **deployed})
                pred_cal, acc_cal = apply_policy(cal['p'], original)
                wrong = (pred_cal != cal['y']) & acc_cal
                for fraction, (indices, sizes) in sample_indices.items():
                    ks = np.column_stack([wrong[ii].sum(axis=1) for ii in indices])
                    limits = np.array([k_max(n, .05/96) for n in sizes])
                    passes = (ks <= limits).all(axis=1)
                    subsamples.append({**base, 'calibration_fraction': fraction,
                                       'repetitions': 500, 'random_seed': 20261027,
                                       'replacement': False, 'rule_reselected': False,
                                       'class_delta': .05/96,
                                       'pass_count': int(passes.sum()), 'pass_rate': float(passes.mean()),
                                       **{f'n_{name}': sizes[c] for c, name in enumerate(CLASSES)},
                                       **{f'k_max_{name}': int(limits[c]) for c, name in enumerate(CLASSES)}})
                mechanism.append({**base, 'threshold_min': min(original['thresholds']),
                                  'threshold_max': max(original['thresholds']),
                                  'temperature': original['temperature'],
                                  'margin': original['margin'],
                                  'calibration_newcastle_n': int((cal['y'] == 3).sum()),
                                  'calibration_newcastle_errors': int((wrong & (cal['y'] == 3)).sum()),
                                  'calibration_newcastle_UB_primary': certificates[fmt][rule]['class_bounds'][3]['upper_bound']})
            print(f'C4 {backbone} seed={seed} {fmt}: ablations complete', flush=True)
    summary = []
    factors = list(dict.fromkeys((r['factor'], r['variant']) for r in details))
    for factor, variant in factors:
        subset = [r for r in details if r['factor'] == factor and r['variant'] == variant]
        passing = [r for r in subset if r['posthoc_passes']]
        summary.append({'analysis': LABEL, 'factor': factor, 'variant': variant,
                        'n_original_slots': len(subset), 'posthoc_pass_count': len(passing),
                        'passing_test_coverage_min': min((r['coverage'] for r in passing), default=None),
                        'passing_test_coverage_max': max((r['coverage'] for r in passing), default=None),
                        'passing_test_worst_error_min': min((r['worst_class_automatic_error'] for r in passing), default=None),
                        'passing_test_worst_error_max': max((r['worst_class_automatic_error'] for r in passing), default=None),
                        'primary_results_replaced': False})
    power = []
    for n in (84, 100, 150, 200, 300, 500):
        for accounting, d in [('primary_96_bounds', .05/96), ('IUT_24_rules', .05/24)]:
            maximum = k_max(n, d)
            for risk in (.005, .01, .02, .03, .05):
                power.append({'analysis': LABEL, 'n_newcastle': n, 'true_newcastle_risk': risk,
                              'accounting': accounting, 'class_delta': d, 'k_max': maximum,
                              'probability_pass_Newcastle': float(binom.cdf(maximum, n, risk)),
                              'scope': 'Newcastle_bound_only; full_rule_also_requires_other_classes'})
    ctx.table('C4_ablation_details', details)
    ctx.table('C4_ablation_summary', summary)
    ctx.table('C4_calibration_subsampling', subsamples)
    ctx.table('C4_calibration_power', power)
    ctx.table('C4_backbone_failure_mechanism', mechanism)
    primary = next(r for r in summary if r['factor'] == 'multiplicity' and r['variant'] == 'primary_96_bounds')
    if primary['posthoc_pass_count'] != 7:
        raise ValueError('Primary ablation reconstruction differs from 7/24; stop')


def run(ctx):
    comparators(ctx)
    ablations(ctx)
    ctx.write_json('C_methodology.json', {
        'analysis': LABEL,
        'primary_result': '7/24 remains unchanged; all new procedures post-hoc',
        'guarantee_interpretation': 'conditional mathematical procedure properties only; family correction cannot make a posthoc design prospective',
        'exploratory_high_probability_family': EXPLORATORY_FAMILY,
        'family_definition': '12 model/seed/precision cases x 6 scores x (CP pipeline and LTT)',
        'CP_class_delta': 'procedure_delta/4 for simultaneous four-class bounds',
        'LTT_rule_delta': 'procedure_delta with within-rule IUT and fixed-sequence stopping',
        'CRC': 'per-class (n*Rhat+1)/(n+1)<=alpha; max of class thresholds; expected risk only',
        'CRC_ties': 'threshold nextafter(error_score,+infinity) for >= acceptance',
        'score_temperature': 'only MSP_temperature uses stored validation temperature; other scores raw p/logits',
        'doctor': 'negative D_alpha=-g/(1-g), g=1-sum(p^2); no perturbation',
        'energy': 'T*logsumexp(logits/T), T=1; stored dequantized INT8 logits',
        'AURC': 'discrete mean of rankwise risks 1..N, exact random tie expectation; discrete oracle subtraction',
        'grid': '50 validation score quantiles including endpoints, deduplicated, strict to permissive',
        'CP_selection_grid_order': 'permissive to strict, first candidate wins equal validation coverage, matching original global search',
        'ablation_search_budget': 'primary vs 21 quantiles and 13 margins 0..0.3 step 0.025',
        'subsampling': '500 common samples without replacement within each class; fixed primary selected rules',
        'ablation_guarantee_warning': 'sensitivity variants are separate post-hoc scenarios, no joint new confirmatory family',
        'excluded_methods': 'SelectiveNet, Deep Gamblers, OSP require retraining or different target/protocol',
    })
