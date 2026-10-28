"""Phase A recomputes descriptive tables from immutable saved artifacts."""
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
from poultryrisk.risk import apply_policy, cp_interval, cp_upper
from .common import CLASSES, EVIDENCE, class_counts, summarize


def candidate_table(ctx):
    rows = []
    for backbone, seed, run in ctx.runs():
        certs = ctx.read_json(run/'certificates.json')
        for fmt in ctx.cfg['formats']:
            test = ctx.predictions(run, fmt, 'test')
            cal = ctx.predictions(run, fmt, 'calibration')
            for rule in ['global', 'proposed']:
                cert = certs[fmt][rule]
                pred, accept = apply_policy(test['p'], cert['deployed_policy'])
                stats = class_counts(test['y'], pred, accept)
                pc, ac = apply_policy(cal['p'], cert['selected_policy'])
                for c, (bound, stat) in enumerate(zip(cert['class_bounds'], stats)):
                    errors = int(((cal['y'] == c) & ac & (pc != cal['y'])).sum())
                    if errors != bound['errors'] or int((cal['y'] == c).sum()) != bound['n']:
                        raise ValueError('Saved certificate/calibration predictions disagree')
                    rows.append({'backbone': backbone, 'seed': seed, 'format': fmt, 'rule': rule,
                                 'class': bound['class'], 'n_cal': bound['n'], 'errors_cal': bound['errors'],
                                 'upper_bound': bound['upper_bound'], 'passes': bound['passes'],
                                 'status': cert['status'], 'n_test': stat['n'], 'accepted_test': stat['accepted'],
                                 'errors_test': stat['errors'], 'coverage_test': stat['coverage'],
                                 'automatic_error_test': stat['automatic_error'],
                                 'selective_error_test': stat['selective_error'], 'evidence': EVIDENCE})
    ctx.table('A1_candidate_class_certificates', rows)


def minimum_n(k, alpha, delta):
    low, high = k+1, max(k+2, 32)
    while cp_upper(k, high, delta) > alpha:
        high *= 2
    while low < high:
        mid = (low+high)//2
        if cp_upper(k, mid, delta) <= alpha:
            high = mid
        else:
            low = mid+1
    return low


def sensitivity(ctx):
    levels = [('uncorrected', 0.05), ('four_classes', 0.05/4),
              ('IUT_24_rules', 0.05/24), ('primary_96_bounds', 0.05/96)]
    rows = [{'accounting': name, 'delta_per_bound': delta, 'n': 84, 'errors': k,
             'upper_bound': cp_upper(k, 84, delta), 'passes_alpha_010': cp_upper(k, 84, delta) <= 0.1,
             'evidence': EVIDENCE} for name, delta in levels for k in range(6)]
    ctx.table('A2_cp_newcastle_sensitivity', rows)
    rows = [{'accounting': name, 'delta_per_bound': delta, 'alpha': alpha, 'errors': k,
             'minimum_n': minimum_n(k, alpha, delta), 'evidence': EVIDENCE}
            for name, delta in levels for alpha in [0.05, 0.10, 0.15, 0.20] for k in range(4)]
    ctx.table('A2_cp_minimum_sample_size', rows)
    rows = []
    for backbone, seed, run in ctx.runs():
        certs = ctx.read_json(run/'certificates.json')
        for fmt in ctx.cfg['formats']:
            for rule in ['global', 'proposed']:
                cert = certs[fmt][rule]
                if cert['status'] == 'CERTIFIED':
                    continue
                bound = cert['class_bounds'][3]
                lo, hi = cp_interval(bound['errors'], bound['n'])
                rows.append({'backbone': backbone, 'seed': seed, 'format': fmt, 'rule': rule,
                             'class': 'newcastle', 'n_cal': bound['n'], 'errors_cal': bound['errors'],
                             'point_estimate': bound['errors']/bound['n'], 'ci95_lower': lo, 'ci95_upper': hi,
                             'primary_status': cert['status'], 'evidence': EVIDENCE})
    ctx.table('A2_failed_newcastle_intervals', rows)


def h2(ctx):
    run = ctx.root/'mobilenetv2'/'seed_29'
    selected = ctx.read_json(run/'selected_policies.json')
    certs = ctx.read_json(run/'certificates.json')
    thresholds, counts, margins = [], [], []
    for fmt in ctx.cfg['formats']:
        for rule in ['global', 'proposed']:
            policy = selected[fmt][rule]
            for c, label in enumerate(CLASSES):
                thresholds.append({'backbone': 'mobilenetv2', 'seed': 29, 'format': fmt, 'rule': rule,
                                   'class': label, 'threshold': policy['thresholds'][c],
                                   'temperature': policy['temperature'], 'margin': policy['margin'],
                                   'primary_status': certs[fmt][rule]['status'], 'evidence': EVIDENCE})
            for split in ['val', 'calibration', 'test']:
                data = ctx.predictions(run, fmt, split)
                for version in ['selected', 'deployed']:
                    chosen = certs[fmt][rule][version+'_policy']
                    pred, accept = apply_policy(data['p'], chosen)
                    for stat in class_counts(data['y'], pred, accept):
                        counts.append({'backbone': 'mobilenetv2', 'seed': 29, 'format': fmt,
                                       'rule': rule, 'split': split, 'policy_version': version,
                                       **stat, 'evidence': EVIDENCE})
        val = ctx.predictions(run, fmt, 'val')
        for margin in [0, 0.05, 0.10, 0.20]:
            policy = copy.deepcopy(selected[fmt]['proposed'])
            policy['margin'] = margin
            pred, accept = apply_policy(val['p'], policy)
            overall = summarize(val['y'], pred, accept)
            for stat in class_counts(val['y'], pred, accept):
                margins.append({'backbone': 'mobilenetv2', 'seed': 29, 'format': fmt,
                                'margin': margin, 'fixed_thresholds': json.dumps(policy['thresholds']),
                                'overall_coverage': overall['coverage'], **stat,
                                'recomputation': 'validation only, selected quantile thresholds held fixed',
                                'evidence': EVIDENCE})
    ctx.table('A3_h2_thresholds', thresholds)
    ctx.table('A3_h2_split_counts', counts)
    ctx.table('A3_h2_margin_recomputation', margins)


def transfer(ctx):
    rows, raw = [], []
    for backbone, seed, run in ctx.runs():
        selected = ctx.read_json(run/'selected_policies.json')
        certs = ctx.read_json(run/'certificates.json')
        policies = ctx.read_json(run/'policies.json')
        data = ctx.predictions(run, 'int8', 'test')
        variants = [('fp32_rule_transferred_unverified', policies['int8']['fp32_rule_transferred_unverified'], 'UNVERIFIED_TRANSFER'),
                    ('int8_selected_proposed', selected['int8']['proposed'], 'UNVERIFIED_AS_SELECTED'),
                    ('int8_deployed_proposed', certs['int8']['proposed']['deployed_policy'], certs['int8']['proposed']['status'])]
        for variant, policy, status in variants:
            pred, accept = apply_policy(data['p'], policy)
            overall = summarize(data['y'], pred, accept)
            for stat in class_counts(data['y'], pred, accept):
                rows.append({'backbone': backbone, 'seed': seed, 'format': 'int8', 'variant': variant,
                             'status': status, **stat, 'overall_accepted': overall['accepted'],
                             'overall_errors': overall['errors'], 'overall_coverage': overall['coverage'],
                             'overall_selective_error': overall['selective_error'],
                             'int8_primary_status': certs['int8']['proposed']['status'],
                             'evidence': 'post-test diagnostic; '+EVIDENCE})
        for fmt in ctx.cfg['formats']:
            test = ctx.predictions(run, fmt, 'test')
            pred = test['p'].argmax(axis=1)
            for c in range(1, 4):
                n = int((test['y'] == c).sum())
                k = int(((test['y'] == c) & (pred == 0)).sum())
                raw.append({'backbone': backbone, 'seed': seed, 'format': fmt, 'class': CLASSES[c],
                            'n': n, 'false_healthy': k, 'false_healthy_fraction': k/n,
                            'scope': 'raw prediction, no abstention', 'evidence': EVIDENCE})
    ctx.table('A4_fp32_to_int8_transfer', rows)
    ctx.table('A4_raw_false_healthy', raw)


def subgroups(ctx):
    rows = []
    for backbone, seed, run in ctx.runs():
        certs = ctx.read_json(run/'certificates.json')
        for fmt in ctx.cfg['formats']:
            saved = ctx.read_json(run/f'evaluation_{fmt}/source_subgroups.json')
            test = ctx.predictions(run, fmt, 'test')
            for rule in ['global', 'proposed']:
                cert = certs[fmt][rule]
                if cert['status'] != 'CERTIFIED':
                    continue
                pred, accept = apply_policy(test['p'], cert['deployed_policy'])
                for source, methods in saved['groups'].items():
                    mask = test['source'] == source
                    stats = class_counts(test['y'][mask], pred[mask], accept[mask])
                    original = methods[rule+'_certified']['classwise']
                    for stat, saved_stat in zip(stats, original):
                        if stat['n'] != saved_stat['n'] or stat['accepted'] != saved_stat['n_accepted']:
                            raise ValueError('Saved subgroup count mismatch')
                        lo, hi = cp_interval(stat['errors'], stat['n'])
                        rows.append({'backbone': backbone, 'seed': seed, 'format': fmt, 'rule': rule,
                                     'source_proxy': source, **stat, 'ci95_lower': lo, 'ci95_upper': hi,
                                     'small_denominator_lt30': stat['n'] < 30,
                                     'status': cert['status'], 'scope': 'descriptive mixed-holdout subgroup; no source-specific guarantee',
                                     'evidence': EVIDENCE})
    ctx.table('A5_source_class_subgroups', rows)


def operator_inspection(ctx):
    """Inspect saved graph tensors/operators, never call invoke or benchmark."""
    cache = ctx.out/'inspection/operator_inspection.json'
    if not cache.exists():
        sibling = ctx.root.parent/'revision_ijies_20267101'/'inspection/operator_inspection.json'
        if sibling.exists() and sibling != cache:
            cache = sibling
    if ctx.allow_missing_models and cache.exists():
        obj = ctx.read_json(cache)
        # Every cached model digest must agree with frozen export metadata.
        for row in obj['models']:
            path = ctx.root/row['backbone']/f"seed_{row['seed']}"
            expected = ctx.read_json(path/f"export_{row['format']}.json")['sha256']
            if expected != row['model_sha256']:
                raise ValueError('Cached operator inspection does not match model metadata')
        return obj
    models, ops = [], []
    try:
        import tensorflow as tf
        if tf.__version__ != '2.20.0':
            raise RuntimeError('Inspection requires TensorFlow 2.20.0, got '+tf.__version__)
        reason = None
    except (ImportError, RuntimeError) as exc:
        tf, reason = None, str(exc)
    for backbone, seed, run in ctx.runs():
        for fmt in ctx.cfg['formats']:
            exported = ctx.read_json(run/f'export_{fmt}.json')
            base = {'backbone': backbone, 'seed': seed, 'format': fmt, 'model_sha256': exported['sha256'],
                    'scope': 'static graph inspection only; no timing or invoke', 'evidence': EVIDENCE}
            if tf is None:
                models.append({**base, 'status': 'UNAVAILABLE', 'reason': reason})
                continue
            path = ctx.track(run/f'model_{fmt}.tflite')
            interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=2,
                experimental_delegates=[], experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
            interpreter.allocate_tensors()
            tensors = {r['index']: r for r in interpreter.get_tensor_details()}
            details = interpreter._get_ops_details()
            float_ops = 0
            quantize = dequantize = 0
            for index, op in enumerate(details):
                inputs = [str(np.dtype(tensors[int(i)]['dtype'])) for i in op['inputs'] if int(i) >= 0 and int(i) in tensors]
                outputs = [str(np.dtype(tensors[int(i)]['dtype'])) for i in op['outputs'] if int(i) >= 0 and int(i) in tensors]
                uses_float = any(name.startswith('float') for name in inputs+outputs)
                float_ops += int(uses_float)
                quantize += int(op['op_name'] == 'QUANTIZE')
                dequantize += int(op['op_name'] == 'DEQUANTIZE')
                ops.append({**base, 'operator_index': index, 'operator_name': op['op_name'],
                            'input_tensor_dtypes': ','.join(inputs), 'output_tensor_dtypes': ','.join(outputs),
                            'uses_float_tensor': uses_float})
            dtype_counts = {}
            for tensor in tensors.values():
                dtype = str(np.dtype(tensor['dtype']))
                dtype_counts[dtype] = dtype_counts.get(dtype, 0)+1
            models.append({**base, 'status': 'INSPECTED', 'reason': None, 'tensorflow_version': tf.__version__,
                           'operators': len(details), 'float_tensor_operators': float_ops,
                           'quantize_operators': quantize, 'dequantize_operators': dequantize,
                           'tensor_dtype_counts': json.dumps(dtype_counts, sort_keys=True)})
    obj = {'models': models, 'operators': ops,
           'limitation': 'Tensor dtypes and graph operators do not prove hardware execution or device latency.'}
    ctx.write_json('inspection/operator_inspection.json', obj)
    return obj


def latency(ctx):
    rows = []
    ctx.read_json('environment.json')
    for backbone, seed, run in ctx.runs():
        for fmt in ctx.cfg['formats']:
            exported = ctx.read_json(run/f'export_{fmt}.json')
            for split in ['val', 'calibration', 'test']:
                saved = ctx.read_json(run/f'latency_{fmt}_{split}.json')
                rows.append({'backbone': backbone, 'seed': seed, 'format': fmt, 'split': split,
                             'n_invocations': int((ctx.manifest.split == split).sum()),
                             'median_invoke_ms': saved['median_invoke_ms'], 'p95_invoke_ms': saved['p95_invoke_ms'],
                             'warmup_invocations': saved['warmup_invocations'], 'threads': saved['threads'],
                             'input_dtype': exported['input_dtype'], 'output_dtype': exported['output_dtype'],
                             'input_scale': exported['input_quantization'][0], 'input_zero_point': exported['input_quantization'][1],
                             'output_scale': exported['output_quantization'][0], 'output_zero_point': exported['output_quantization'][1],
                             'model_size_bytes': exported['size_bytes'],
                             'host_cpu_model': 'not recorded', 'raw_per_image_timings': 'not saved',
                             'scope': 'original Colab host CPU invoke only; loading/preprocessing excluded; no new benchmark',
                             'evidence': EVIDENCE})
    ctx.table('A6_saved_host_latency', rows)
    inspection = operator_inspection(ctx)
    models, operators = operator_table_rows(inspection)
    ctx.table('A6_operator_models', models)
    ctx.table('A6_operator_details', operators)
    grouped = []
    for row in inspection['models']:
        if row['status'] != 'INSPECTED':
            continue
        keys = ['backbone', 'seed', 'format']
        selected = [x for x in inspection['operators'] if all(x[k] == row[k] for k in keys)]
        counts = {}
        for op in selected:
            key = (op['operator_name'], op['input_tensor_dtypes'], op['output_tensor_dtypes'], op['uses_float_tensor'])
            counts[key] = counts.get(key, 0)+1
        for (name, inp, out, uses_float), n in sorted(counts.items()):
            grouped.append({**{k: row[k] for k in keys}, 'operator_name': name, 'input_tensor_dtypes': inp,
                            'output_tensor_dtypes': out, 'uses_float_tensor': uses_float, 'count': n, 'evidence': EVIDENCE})
    ctx.table('A6_operator_dtype_counts', grouped)


def operator_table_rows(inspection):
    """Keep CSV schemas identical after a JSON cache round trip.

    JSON cache keys are sorted for reproducibility. DataFrame construction must
    not inherit that ordering in place of the original native inspection schema.
    """
    base = ['backbone', 'seed', 'format', 'model_sha256', 'scope', 'evidence']
    models = base+['status', 'reason', 'tensorflow_version', 'operators',
                   'float_tensor_operators', 'quantize_operators', 'dequantize_operators',
                   'tensor_dtype_counts']
    operators = base+['operator_index', 'operator_name', 'input_tensor_dtypes',
                      'output_tensor_dtypes', 'uses_float_tensor']
    def ordered(rows, columns):
        return [{key: row[key] for key in columns if key in row} for row in rows]
    return ordered(inspection['models'], models), ordered(inspection['operators'], operators)


def run(ctx):
    for name, fn in [('A1', candidate_table), ('A2', sensitivity), ('A3', h2), ('A4', transfer), ('A5', subgroups), ('A6', latency)]:
        print('Phase '+name, flush=True)
        fn(ctx)
