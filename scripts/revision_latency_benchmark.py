"""Post-hoc host latency benchmark of the frozen TFLite exports on a documented CPU.

Models and saved test predictions are read from results/full_colab_v1 without
writing there; poultryrisk.models.infer is never called. Labels are not read and
no accuracy metric is computed. Host CPU only: not smartphone or device latency.
"""
import argparse
import io
import json
import platform
import resource
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poultryrisk.common import digest, read_json
from poultryrisk.models import read_image

RUN = ROOT/'results/full_colab_v1'
OUT = ROOT/'results/revision_ijies_20267101/latency_host_documented'
SEED = 20261027
WARMUP = 50
PASSES = 3


def now():
    return datetime.now(timezone.utc).isoformat()


def rss_kib():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1])
    return None


def windows_load():
    try:
        res = subprocess.run(['powershell.exe', '-NoProfile', '-Command',
                              '(Get-CimInstance Win32_Processor).LoadPercentage'],
                             capture_output=True, text=True, timeout=60)
        return int(res.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def test_frame():
    manifest = pd.read_csv(RUN/'data/manifest.csv')
    return manifest[manifest.split == 'test'].sort_values('image_id')[['image_id', 'path']]


def worker(args):
    import tensorflow as tf
    cfg = read_json(RUN/'experiment_freeze.json')['config']
    run = RUN/args.backbone/f'seed_{args.seed}'
    model = run/f'model_{args.fmt}.tflite'
    if digest(model) != read_json(run/f'export_{args.fmt}.json')['sha256']:
        raise ValueError('TFLite model hash mismatch')
    frame = test_frame()
    with zipfile.ZipFile(ROOT/'archive.zip') as z:
        blobs = [z.read(p) for p in frame.path]
    rss_before = rss_kib()
    interpreter = tf.lite.Interpreter(model_path=str(model), num_threads=args.threads)
    interpreter.allocate_tensors()
    rss_allocated = rss_kib()
    inp, op = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    in_scale, in_zero = inp['quantization']
    out_scale, out_zero = op['quantization']
    integer_in = np.issubdtype(inp['dtype'], np.integer)
    integer_out = np.issubdtype(op['dtype'], np.integer)
    limits = np.iinfo(inp['dtype']) if integer_in else None

    def prepare(blob):
        # Same input conversion as poultryrisk.models.infer.
        x = read_image(io.BytesIO(blob), cfg['image_size'])[None, ...]
        if integer_in:
            return np.clip(np.rint(x/in_scale+in_zero), limits.min, limits.max).astype(inp['dtype'])
        return x.astype(inp['dtype'])

    interpreter.set_tensor(inp['index'], prepare(blobs[0]))
    for _ in range(WARMUP):
        interpreter.invoke()
    invoke = np.zeros((PASSES, len(blobs)), dtype=np.int64)
    end_to_end = np.zeros_like(invoke)
    logits = []
    started = now()
    for k in range(PASSES):
        for i, blob in enumerate(blobs):
            t0 = time.perf_counter_ns()
            interpreter.set_tensor(inp['index'], prepare(blob))
            t1 = time.perf_counter_ns()
            interpreter.invoke()
            t2 = time.perf_counter_ns()
            raw = interpreter.get_tensor(op['index'])[0]
            if integer_out:
                raw = (raw.astype(np.float32)-out_zero)*out_scale
            e = np.exp(raw-raw.max())
            int((e/e.sum()).argmax())
            t3 = time.perf_counter_ns()
            invoke[k, i], end_to_end[k, i] = t2-t1, t3-t0
            if k == 0:
                logits.append(raw)
    finished = now()
    logits = np.asarray(logits)
    saved = np.load(run/f'predictions_{args.fmt}_test.npz')
    order = {v: j for j, v in enumerate(saved['image_id'])}
    ref = saved['logits'][[order[v] for v in frame.image_id]]
    same_rows = np.all(logits == ref, axis=1)
    np.savez_compressed(args.out.with_suffix('.npz'), image_id=frame.image_id.to_numpy(dtype=str),
                        invoke_ns=invoke, end_to_end_ns=end_to_end)
    args.out.write_text(json.dumps({
        'backbone': args.backbone, 'seed': args.seed, 'format': args.fmt, 'threads': args.threads,
        'model_sha256': digest(model), 'n_images': len(blobs), 'warmup_invocations': WARMUP, 'passes': PASSES,
        'started_utc': started, 'finished_utc': finished,
        'rss_before_interpreter_kib': rss_before, 'rss_after_allocate_kib': rss_allocated,
        'interpreter_rss_increment_kib': rss_allocated-rss_before,
        'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'output_identical_fraction': float(same_rows.mean()),
        'output_max_abs_diff': float(np.abs(logits-ref).max()),
        'argmax_agreement': float((logits.argmax(1) == ref.argmax(1)).mean())}, indent=2))


def host_info():
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception:
            return None
    import PIL
    import tensorflow as tf
    flags = Path('/proc/cpuinfo').read_text().split('flags')[1].split('\n')[0].split()
    return {'lscpu': sh(['lscpu']), 'kernel': platform.release(),
            'isa_flags': {f: f in flags for f in ['sse4_2', 'avx', 'avx2', 'fma', 'avx512f', 'avx512_vnni', 'avx_vnni']},
            'meminfo_total': Path('/proc/meminfo').read_text().splitlines()[0],
            'windows': sh(['powershell.exe', '-NoProfile', '-Command',
                           "$o=Get-CimInstance Win32_OperatingSystem; $p=Get-CimInstance Win32_Processor; "
                           "\"$($o.Caption) $($o.Version); $($p.Name); cores $($p.NumberOfCores); "
                           "logical $($p.NumberOfLogicalProcessors)\"; powercfg /getactivescheme"]),
            'python': platform.python_version(), 'tensorflow': tf.__version__, 'numpy': np.__version__,
            'pillow': PIL.__version__, 'interpreter': 'tf.lite.Interpreter (TF 2.20.0, default delegates)',
            'colab_lock_versions': [l for l in (ROOT/'results/colab_dependencies.lock.txt').read_text().splitlines()
                                    if l.lower().split('==')[0] in ('tensorflow', 'numpy', 'pillow')]}


def summarize(records):
    rows = []
    for rec in records:
        data = np.load(Path(rec['raw_file']))
        colab = read_json(RUN/rec['backbone']/f"seed_{rec['seed']}"/f"latency_{rec['format']}_test.json")
        for measure, key in [('invoke', 'invoke_ns'), ('end_to_end', 'end_to_end_ns')]:
            ms = data[key].astype(np.float64)/1e6
            flat = ms.ravel()
            q1, med, q3, p95 = np.quantile(flat, [0.25, 0.5, 0.75, 0.95])
            rows.append({'backbone': rec['backbone'], 'seed': rec['seed'], 'format': rec['format'],
                         'threads': rec['threads'], 'measure': measure, 'n_timings': flat.size,
                         'median_ms': med, 'q1_ms': q1, 'q3_ms': q3, 'iqr_ms': q3-q1, 'p95_ms': p95,
                         'mean_ms': flat.mean(), 'sd_ms': flat.std(ddof=1), 'min_ms': flat.min(), 'max_ms': flat.max(),
                         'pass_medians_ms': '/'.join(f'{np.median(r):.3f}' for r in ms),
                         'colab_median_invoke_ms_2threads': colab['median_invoke_ms'] if measure == 'invoke' else None,
                         'colab_p95_invoke_ms_2threads': colab['p95_invoke_ms'] if measure == 'invoke' else None,
                         'interpreter_rss_increment_mib': rec['interpreter_rss_increment_kib']/1024,
                         'peak_process_rss_mib': rec['peak_rss_kib']/1024,
                         'windows_cpu_load_before_pct': rec['windows_load_before'],
                         'windows_cpu_load_after_pct': rec['windows_load_after']})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--backbone')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--fmt')
    parser.add_argument('--threads', type=int)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f'{OUT} is not empty; use a new folder rather than overwriting results')
    (OUT/'raw').mkdir(parents=True, exist_ok=True)
    started = now()
    (OUT/'host_info.json').write_text(json.dumps(host_info(), indent=2))
    jobs = [(b, s, f, t) for b in ['mobilenetv2', 'efficientnetb0'] for s in [17, 29, 43]
            for f in ['fp32', 'int8'] for t in [2, 1]]
    order = np.random.default_rng(SEED).permutation(len(jobs))
    records = []
    log = (OUT/'run.log').open('a')
    for rank, j in enumerate(order, 1):
        b, s, f, t = jobs[j]
        out = OUT/'raw'/f'{b}_s{s}_{f}_t{t}.json'
        load_before = windows_load()
        cmd = [sys.executable, '-I', str(Path(__file__).resolve()), '--worker', '--backbone', b,
               '--seed', str(s), '--fmt', f, '--threads', str(t), '--out', str(out)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        load_after = windows_load()
        log.write(f'{now()} job {rank}/{len(jobs)} {b} s{s} {f} t{t} exit {res.returncode} '
                  f'win_load_before {load_before} after {load_after}\n')
        log.flush()
        if res.returncode:
            log.write(res.stderr[-4000:]+'\n')
            raise SystemExit(f'worker failed: {b} s{s} {f} t{t}')
        rec = json.loads(out.read_text())
        rec.update(raw_file=str(out.with_suffix('.npz')), run_rank=rank,
                   windows_load_before=load_before, windows_load_after=load_after)
        records.append(rec)
        print(f'{rank}/{len(jobs)} {b} s{s} {f} t{t} done', flush=True)
    summary = summarize(records)
    summary.to_csv(OUT/'summary.csv', index=False)
    checks = pd.DataFrame([{k: r[k] for k in ['backbone', 'seed', 'format', 'threads', 'model_sha256',
                                               'output_identical_fraction', 'output_max_abs_diff', 'argmax_agreement']}
                           for r in records])
    checks.to_csv(OUT/'output_check.csv', index=False)
    parts = []
    for r in records:
        data = np.load(Path(r['raw_file']))
        for k in range(PASSES):
            parts.append(pd.DataFrame({'backbone': r['backbone'], 'seed': r['seed'], 'format': r['format'],
                                       'threads': r['threads'], 'pass': k+1, 'image_id': data['image_id'],
                                       'invoke_ns': data['invoke_ns'][k], 'end_to_end_ns': data['end_to_end_ns'][k]}))
    pd.concat(parts).to_csv(OUT/'raw_timings.csv.gz', index=False)
    (OUT/'run_info.json').write_text(json.dumps({
        'started_utc': started, 'finished_utc': now(), 'order_seed': SEED,
        'job_order': [list(jobs[j]) for j in order], 'warmup_invocations': WARMUP, 'passes': PASSES,
        'inputs': '1,154 frozen test images, JPEG bytes preloaded from archive.zip; labels not read',
        'invoke_scope': 'interpreter.invoke() only, as in the original Colab measurement',
        'end_to_end_scope': 'in-memory JPEG decode, resize, input conversion, set_tensor, invoke, '
                            'get_tensor, dequantization, softmax and argmax; excludes file I/O',
        'memory_scope': 'Linux process RSS on the host; not device memory',
        'evidence': 'post-hoc host benchmark; original Colab latency and all certificates unchanged'}, indent=2))
    files = sorted(p for p in OUT.rglob('*') if p.is_file() and p.name != 'SHA256SUMS.txt')
    (OUT/'SHA256SUMS.txt').write_text(''.join(f'{digest(p)}  {p.relative_to(OUT).as_posix()}\n' for p in files))
    print('benchmark complete', flush=True)


if __name__ == '__main__':
    main()
