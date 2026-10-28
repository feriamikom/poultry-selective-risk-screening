"""Post-hoc image similarity, paired uncertainty and freeze provenance.

No image is treated as a bird/flock; bootstrap intervals condition on the
retained-image sample and do not establish independence across birds/farms.
"""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import hashlib
import io
import json
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

from poultryrisk.risk import apply_policy
from .common import CLASSES

SEED = 20261027
POPCOUNT = np.array([int(x).bit_count() for x in range(256)], dtype=np.uint8)
LABEL = 'post-hoc descriptive; retained images, not verified independent birds/flocks'


def paired_bootstrap(y, pred_a, accept_a, pred_b, accept_b,
                     repetitions=2000, seed=SEED):
    """A minus B using identical class-stratified draws, separately per seed.

    The worst-class statistic is recomputed within every bootstrap sample.
    Discordance counts are image counts, including a percentile interval.
    """
    y = np.asarray(y)
    pred_a, pred_b = np.asarray(pred_a), np.asarray(pred_b)
    accept_a, accept_b = np.asarray(accept_a, bool), np.asarray(accept_b, bool)
    if not all(len(x) == len(y) for x in (pred_a, pred_b, accept_a, accept_b)):
        raise ValueError('Paired inputs must have identical length/order')
    if not len(y) or repetitions < 1:
        raise ValueError('Nonempty sample and positive repetitions required')
    error_a, error_b = accept_a & (pred_a != y), accept_b & (pred_b != y)
    discordant = accept_a != accept_b
    rng = np.random.default_rng(seed)
    cov_draw = np.zeros(repetitions)
    discord_draw = np.zeros(repetitions)
    ra, rb, point_a, point_b = [], [], [], []
    rows = []
    for c, name in enumerate(CLASSES):
        ix = np.flatnonzero(y == c)
        if not len(ix):
            continue
        draws = ix[rng.integers(0, len(ix), (repetitions, len(ix)))]
        ac = error_a[draws].mean(axis=1)
        bc = error_b[draws].mean(axis=1)
        ra.append(ac)
        rb.append(bc)
        point_a.append(float(error_a[ix].mean()))
        point_b.append(float(error_b[ix].mean()))
        cov_draw += (accept_a[draws].astype(float)-accept_b[draws]).sum(axis=1)/len(y)
        discord_draw += discordant[draws].sum(axis=1)
        rows.append(_interval('automatic_error_'+name,
                             point_a[-1]-point_b[-1], ac-bc))
    rows.insert(0, _interval('coverage', float(accept_a.mean()-accept_b.mean()), cov_draw))
    rows.append(_interval('worst_class_automatic_error', max(point_a)-max(point_b),
                          np.max(ra, axis=0)-np.max(rb, axis=0)))
    rows.append(_interval('discordant_count', int(discordant.sum()), discord_draw))
    return rows


def _interval(metric, estimate, samples):
    low, high = np.quantile(samples, [0.025, 0.975])
    return {'metric': metric, 'estimate': estimate,
            'ci_low': float(low), 'ci_high': float(high)}


def _hashes(frame):
    return np.array([int(str(x), 16) for x in frame.phash], dtype=np.uint64)


def nearest_hamming(query, reference, same=False, block=96, collect_pairs=False):
    """64-bit exact Hamming nearest neighbours; exclude only self, not ties."""
    q, r = np.asarray(query, np.uint64), np.asarray(reference, np.uint64)
    if same and not np.array_equal(q, r):
        raise ValueError('same=True requires identical ordered arrays')
    if len(r) < (2 if same else 1):
        raise ValueError('Insufficient reference images')
    minima, indices, pairs = [], [], []
    for start in range(0, len(q), block):
        stop = min(start+block, len(q))
        xor = np.ascontiguousarray(q[start:stop, None] ^ r[None, :])
        distances = POPCOUNT[xor.view(np.uint8).reshape(stop-start, len(r), 8)].sum(axis=2)
        if same:
            distances[np.arange(stop-start), np.arange(start, stop)] = 255
        arg = distances.argmin(axis=1)
        minima.extend(distances[np.arange(stop-start), arg].tolist())
        indices.extend(arg.tolist())
        if collect_pairs:
            qi, ri = np.where((distances >= 5) & (distances <= 8))
            pairs.extend((int(a+start), int(b), int(distances[a,b])) for a,b in zip(qi,ri))
    return np.asarray(minima, int), np.asarray(indices, int), pairs


def cluster_tables(ctx):
    audit = ctx.read_csv('data/audit_all.csv')
    frame = ctx.manifest.sort_values('image_id').reset_index(drop=True)
    groups = audit.groupby('group_id', sort=True)
    size_rows = []
    for size, count in groups.size().value_counts().sort_index().items():
        size_rows.append({'group_size': int(size), 'groups': int(count),
                          'images': int(size*count), 'analysis': LABEL})
    ctx.table('B1_group_size_distribution', size_rows)
    conflict = groups.label.nunique() > 1
    ctx.table('B1_cleaning_summary', [{
        'input_images': len(audit), 'groups': groups.ngroups,
        'label_conflict_groups': int(conflict.sum()),
        'label_conflict_images': int((audit.decision == 'label_conflict').sum()),
        'ordinary_near_duplicate_images': int((audit.decision == 'near_duplicate').sum()),
        'retained_images': len(frame),
        'retained_groups': int(frame.group_id.nunique()),
        'prior_visual_inspection': 'author-reported deterministic sample: 8 conflict and 8 ordinary near-duplicate groups before training',
        'analysis': LABEL}])
    source_rows = []
    for split in ('train','val','calibration','test'):
        for c,name in enumerate(CLASSES):
            n = int(((frame.split == split) & (frame.y == c)).sum())
            for source in sorted(frame.source.unique()):
                count = int(((frame.split == split) & (frame.y == c) & (frame.source == source)).sum())
                source_rows.append({'split': split, 'class': name, 'source_proxy': source,
                                    'n': count, 'class_split_total': n,
                                    'source_fraction_within_class_split': count/n if n else None,
                                    'analysis': LABEL})
    ctx.table('B1_source_split_class', source_rows)
    train = frame[frame.split == 'train'].reset_index(drop=True)
    near_rows, summaries, histograms = [], [], []
    for split in ('calibration', 'test'):
        print('B1 exact pHash neighbours: '+split, flush=True)
        part = frame[frame.split == split].reset_index(drop=True)
        distances_train, nn_train, _ = nearest_hamming(_hashes(part), _hashes(train))
        distances_same, nn_same, _ = nearest_hamming(_hashes(part), _hashes(part), same=True)
        for i,row in part.iterrows():
            near_rows.append({'image_id': row.image_id, 'class': CLASSES[int(row.y)],
                'split': split, 'source_proxy': row.source,
                'nearest_train_hamming': int(distances_train[i]),
                'nearest_train_image_id': train.iloc[nn_train[i]].image_id,
                'nearest_same_split_hamming': int(distances_same[i]),
                'nearest_same_split_image_id': part.iloc[nn_same[i]].image_id,
                'analysis': LABEL})
        for c,name in [(-1,'ALL')] + list(enumerate(CLASSES)):
            mask = np.ones(len(part),bool) if c == -1 else part.y.to_numpy() == c
            for comparison,distance in [('to_train',distances_train),('within_split',distances_same)]:
                arr = distance[mask]
                summaries.append({'split': split, 'class': name, 'comparison': comparison,
                    'n': len(arr), 'minimum_hamming': int(arr.min()) if len(arr) else None,
                    'median_hamming': float(np.median(arr)) if len(arr) else None,
                    'n_le_4': int((arr<=4).sum()), 'n_le_6': int((arr<=6).sum()),
                    'n_le_8': int((arr<=8).sum()),'n_le_10':int((arr<=10).sum()),
                    'analysis': LABEL})
                for d in range(18):
                    histograms.append({'split':split,'class':name,'comparison':comparison,
                        'hamming_bin': str(d) if d <= 16 else '>16',
                        'n':int((arr==d).sum()) if d<=16 else int((arr>16).sum()),
                        'analysis': LABEL})
    ctx.table('B1_nearest_hamming_per_image', near_rows)
    ctx.table('B1_nearest_hamming_summary', summaries)
    ctx.table('B1_nearest_hamming_histogram', histograms)
    cross_split_review(ctx, frame)


def cross_split_review(ctx, frame):
    """Count the entire requested pHash frame even when image ZIP is absent."""
    eligible=[]
    splits=('train','val','calibration','test')
    for i,left in enumerate(splits):
        a=frame[frame.split==left].reset_index(drop=True)
        for right in splits[i+1:]:
            b=frame[frame.split==right].reset_index(drop=True)
            _,_,pairs=nearest_hamming(_hashes(a),_hashes(b),collect_pairs=True)
            eligible.extend((a.iloc[ai],b.iloc[bi],d) for ai,bi,d in pairs)
    print(f'B1 internal montage: {len(eligible)} eligible cross-split pairs at Hamming5..8; requested24',flush=True)
    ctx.table('B1_cross_split_review_pairs', [{
        'left_image_id':a.image_id,'left_split':a['split'],'left_class':a.label,
        'right_image_id':b.image_id,'right_split':b['split'],'right_class':b.label,
        'hamming':d,'analysis':LABEL} for a,b,d in eligible])
    ctx.table('B1_cross_split_review_sampling', [{
        'requested_pairs':24,'eligible_unique_pairs':len(eligible),
        'selected_without_replacement':min(24,len(eligible)),
        'distance_min_inclusive':5,'distance_max_inclusive':8,
        'sampling_frame':'all retained-image pairs across different frozen splits',
        'limitation':'fewer than24 eligible pairs; show all available' if len(eligible)<24 else None,
        'image_assessment_status':'manual author review required', 'analysis':LABEL}])
    _montage(ctx,eligible)


def _montage(ctx, eligible):
    """Internal author-review only; never required by artifact-only reproduction."""
    archive = Path(__file__).resolve().parents[1]/'archive.zip'
    if not archive.exists():
        return
    ctx.track(archive)
    out = ctx.out/'internal'
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(len(eligible), min(24,len(eligible)), replace=False)
    sheet = Image.new('RGB', (1120, max(1,len(chosen))*172), 'white')
    draw = ImageDraw.Draw(sheet)
    records = []
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        mapping = {}
        for name in names:
            if not name.endswith('/'):
                mapping[name.replace('\\','/')] = name
        for i,index in enumerate(chosen):
            a,b,distance = eligible[int(index)]
            for j,row in enumerate((a,b)):
                path = row.path.replace('\\','/')
                matches = [name for key,name in mapping.items() if key == path or key.endswith('/'+path)]
                if len(matches) != 1:
                    raise ValueError(f'Archive image path has {len(matches)} matches: {path}')
                with Image.open(io.BytesIO(z.read(matches[0]))) as im:
                    thumb = ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'), (156,156))
                sheet.paste(thumb, (j*560, i*172+4))
                draw.text((j*560+165,i*172+5), f'{row["split"]} / {row.label}\n{row.path}\nsource proxy: {row.source}\npHash distance {distance}\nINTERNAL AUTHOR REVIEW ONLY',fill='black')
            records.append({'left_id':a.image_id,'left_path':a.path,'left_split':a['split'],
                            'right_id':b.image_id,'right_path':b.path,'right_split':b['split'],
                            'hamming':distance})
    sheet.save(out/'B1_cross_split_pairs_internal.png')
    (out/'B1_cross_split_pairs_internal.json').write_text(json.dumps({
        'status':'awaiting manual author inspection', 'seed':SEED,
        'sampling_frame':'all distinct retained-image pairs across any two of train/val/calibration/test, Hamming distance 5 to 8 inclusive',
        'requested_pairs':24,'eligible_pairs':len(eligible),'selected_pairs':len(records),'pairs':records,
        'limitation':None if len(records)==24 else 'fewer than24 eligible retained cross-split pairs exist; all available pairs shown without replacement',
        'distribution':'INTERNAL ONLY; images must not be redistributed'},indent=2),encoding='utf-8')


def bootstrap_tables(ctx):
    rows = []
    for backbone,seed,path in ctx.runs():
        print(f'B2 paired bootstrap: {backbone} seed={seed}', flush=True)
        cert = ctx.read_json(path/'certificates.json')
        data = {fmt:ctx.predictions(path,fmt,'test') for fmt in ('fp32','int8')}
        if not np.array_equal(data['fp32']['image_id'],data['int8']['image_id']):
            raise ValueError('FP32/INT8 images must be paired in identical order')
        for deployment in ('as-selected','as-deployed'):
            policy_key = 'selected_policy' if deployment == 'as-selected' else 'deployed_policy'
            for rule in ('global','proposed'):
                a,b = data['int8'],data['fp32']
                pa,aa=apply_policy(a['p'],cert['int8'][rule][policy_key])
                pb,ab=apply_policy(b['p'],cert['fp32'][rule][policy_key])
                base={'backbone':backbone,'seed':seed,'comparison':'INT8 minus FP32',
                      'rule':rule,'format':'paired','deployment':deployment,
                      'policy_a_status':cert['int8'][rule]['status'],
                      'policy_b_status':cert['fp32'][rule]['status']}
                rows.extend({**base,**x,'B':2000,'bootstrap_seed':SEED,'analysis':LABEL}
                            for x in paired_bootstrap(a['y'],pa,aa,pb,ab))
            for fmt in ('fp32','int8'):
                a = data[fmt]
                pa,aa=apply_policy(a['p'],cert[fmt]['proposed'][policy_key])
                pb,ab=apply_policy(a['p'],cert[fmt]['global'][policy_key])
                base={'backbone':backbone,'seed':seed,'comparison':'proposed minus global',
                      'rule':'paired','format':fmt,'deployment':deployment,
                      'policy_a_status':cert[fmt]['proposed']['status'],
                      'policy_b_status':cert[fmt]['global']['status']}
                rows.extend({**base,**x,'B':2000,'bootstrap_seed':SEED,'analysis':LABEL}
                            for x in paired_bootstrap(a['y'],pa,aa,pb,ab))
    ctx.table('B2_paired_bootstrap', rows)


def freeze_tables(ctx):
    """Hash links are substantive; copied file mtimes alone cannot prove chronology."""
    snapshot = Path(__file__).with_name('freeze_export_metadata.json')
    if snapshot.exists():
        metadata = json.loads(ctx.track(snapshot).read_text(encoding='utf-8'))
    else:
        metadata = {'scope':'filesystem mtimes from extraction plus original command logs; neither is an external trusted timestamp attestation',
                    'files':{}, 'stage_execution':{}}
        for _,_,path in ctx.runs():
            for name in ['selection_freeze.json','policy_freeze.json']+[
                    f'predictions_{fmt}_{split}.npz' for fmt in ('fp32','int8') for split in ('calibration','test')]:
                p = path/name
                metadata['files'][p.relative_to(ctx.root).as_posix()] = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()
        session = ctx.root.parent/'session_20260923T013608010722Z'
        for stage,name in [('selection_freeze','full_select_policy_resume'),
                           ('calibration_predictions','full_predict_calibration'),
                           ('policy_freeze','full_calibrate'),('test_predictions','full_evaluate')]:
            p=ctx.track(session/(name+'.json'))
            record=json.loads(p.read_text(encoding='utf-8'))
            metadata['stage_execution'][stage]={'source_relative_path':p.relative_to(ctx.root.parent).as_posix(),
                'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                'started_utc':record['started_utc'],'elapsed_seconds':record['elapsed_seconds'],
                'exit_code':record['exit_code'],
                'finished_utc_derived':(datetime.fromisoformat(record['started_utc'])+timedelta(seconds=record['elapsed_seconds'])).isoformat()}
        snapshot.write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
        ctx.track(snapshot)
    detailed, summary = [], []
    for backbone,seed,path in ctx.runs():
        selection=ctx.read_json(path/'selection_freeze.json')
        policy=ctx.read_json(path/'policy_freeze.json')
        for fmt in ('fp32','int8'):
            rows={}
            for stage,name in [('selection_freeze','selection_freeze.json'),
                    ('calibration_predictions',f'predictions_{fmt}_calibration.npz'),
                    ('policy_freeze','policy_freeze.json'),('test_predictions',f'predictions_{fmt}_test.npz')]:
                p=ctx.track(path/name)
                sha=hashlib.sha256(p.read_bytes()).hexdigest()
                mtime=metadata['files'][p.relative_to(ctx.root).as_posix()]
                rows[stage]=(mtime,sha)
                execution=metadata['stage_execution'][stage]
                detailed.append({'backbone':backbone,'seed':seed,'format':fmt,'stage':stage,
                    'relative_path':p.relative_to(ctx.root).as_posix(),'sha256':sha,
                    'export_mtime_utc':mtime,
                    'internal_created_utc':selection.get('created_utc') if stage=='selection_freeze' else None,
                    'stage_started_utc':execution['started_utc'],
                    'stage_finished_utc_derived':execution['finished_utc_derived'],
                    'stage_exit_code':execution['exit_code'],
                    'stage_log_sha256':execution['source_sha256'],
                    'policy_freeze_tracks_file':name in policy['files'],
                    'evidence_limit':'export timestamps are mutable; hashes establish content linkage, not external temporal attestation',
                    'analysis':'post-hoc provenance audit'})
            times=[rows[x][0] for x in ('selection_freeze','calibration_predictions','policy_freeze','test_predictions')]
            monotone=all(datetime.fromisoformat(a)<=datetime.fromisoformat(b) for a,b in zip(times,times[1:]))
            execution=metadata['stage_execution']
            stages=('selection_freeze','calibration_predictions','policy_freeze','test_predictions')
            ordered=all(datetime.fromisoformat(execution[a]['finished_utc_derived']) <= datetime.fromisoformat(execution[b]['started_utc']) for a,b in zip(stages,stages[1:]))
            summary.append({'backbone':backbone,'seed':seed,'format':fmt,
                'selection_created_utc':selection.get('created_utc'),
                'selection_export_mtime_utc':times[0],'calibration_export_mtime_utc':times[1],
                'policy_export_mtime_utc':times[2],'test_export_mtime_utc':times[3],
                'export_mtimes_in_declared_order':monotone,
                'export_mtime_status':'extraction timestamps; cannot establish historical chronology',
                'stage_log_intervals_in_declared_order':ordered,
                'selection_stage_started_utc':execution['selection_freeze']['started_utc'],
                'calibration_stage_started_utc':execution['calibration_predictions']['started_utc'],
                'policy_stage_started_utc':execution['policy_freeze']['started_utc'],
                'test_stage_started_utc':execution['test_predictions']['started_utc'],
                'selection_calibration_used_flag':selection.get('calibration_used'),
                'selection_test_used_flag':selection.get('test_used'),
                'policy_test_used_flag':policy.get('test_used'),
                'hash_linked_selection_and_calibration':policy['files'].get('selection_freeze.json')==rows['selection_freeze'][1] and policy['files'].get(f'predictions_{fmt}_calibration.npz')==rows['calibration_predictions'][1],
                'evidence_limit':'conditional internal chronology, not an independently timestamped preregistration',
                'analysis':'post-hoc provenance audit'})
    ctx.table('B3_freeze_hashes_timestamps',detailed)
    ctx.table('B3_freeze_order_summary',summary)


def run(ctx):
    print('Phase B1 clustering and source proxies', flush=True)
    cluster_tables(ctx)
    print('Phase B2 paired bootstrap (B=2000, no seed pooling)', flush=True)
    bootstrap_tables(ctx)
    print('Phase B3 hash and chronology audit', flush=True)
    freeze_tables(ctx)
