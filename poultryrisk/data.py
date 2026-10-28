"""Image audit; near-duplicate grouping; frozen class-stratified splits."""
import csv
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.fft import dctn

from .common import digest, read_json, resolve_path, write_json

ALIASES = {'healthy': 'healthy', 'cocci': 'coccidiosis', 'coccidiosis': 'coccidiosis',
           'salmo': 'salmonella', 'salmonella': 'salmonella', 'salmonellosis': 'salmonella',
           'ncd': 'newcastle', 'nd': 'newcastle', 'newcastle': 'newcastle',
           'newcastledisease': 'newcastle'}


def canonical(value):
    return ALIASES.get(re.sub('[^a-z]', '', str(value).lower()))


def filename_info(path):
    token = re.split(r'[._\-\s\d]', Path(path).name.lower())[0]
    source = 'pcr_prefix' if token.startswith('pcr') else 'field_prefix'
    token = token[3:] if token.startswith('pcr') else token
    label = canonical(token)
    if label is None:
        source = 'unknown'
        label = canonical(Path(path).parent.name)
    return label, source


def extract_zip(archive, destination):
    archive, destination = Path(archive), Path(destination).resolve()
    if not archive.is_file():
        raise FileNotFoundError(f'Dataset ZIP missing: {archive}')
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        if sum(i.file_size for i in infos) > 10 * 1024**3:
            raise ValueError('Archive exceeds 10 GiB uncompressed limit')
        for i in infos:
            target = (destination / i.filename.replace('\\', '/')).resolve()
            if not target.is_relative_to(destination) or (i.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Unsafe archive path or symlink')
            if target.exists() and not i.is_dir():
                raise FileExistsError(f'Refusing to overwrite {target}; use an empty data directory')
        z.extractall(destination)


def phash(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert('L').resize((32, 32), Image.Resampling.LANCZOS)
        low = dctn(np.asarray(im, dtype=float), norm='ortho')[:8, :8]
        bits = low > np.median(low)
        value = 0
        for b in bits.ravel():
            value = (value << 1) | int(b)
    return value


class BKTree:
    def __init__(self):
        self.root = None

    def add(self, value, index):
        if self.root is None:
            self.root = [value, index, {}]
            return
        node = self.root
        while True:
            d = (value ^ node[0]).bit_count()
            if d == 0:
                return
            if d not in node[2]:
                node[2][d] = [value, index, {}]
                return
            node = node[2][d]

    def query(self, value, radius):
        stack = [self.root] if self.root else []
        result = []
        while stack:
            v, i, children = stack.pop()
            d = (value ^ v).bit_count()
            if d <= radius:
                result.append(i)
            stack.extend(child for edge, child in children.items() if d-radius <= edge <= d+radius)
        return result


def audit(cfg):
    root = resolve_path(cfg['data_dir'])
    out = resolve_path(cfg['output_dir']) / 'data'
    if (out / 'manifest.csv').exists():
        raise FileExistsError('Frozen manifest exists. Reuse it or choose a NEW output_dir.')
    files = sorted(p for p in root.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'})
    if not files:
        raise FileNotFoundError(f'No images in {root}. Extract dataset first.')
    csv_labels = {}
    for p in root.rglob('*.csv'):
        with p.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            fields = {x.lower().strip(): x for x in (reader.fieldnames or [])}
            namecol = next((fields[x] for x in ['filename', 'file', 'image', 'images', 'image_path'] if x in fields), None)
            labelcol = next((fields[x] for x in ['label', 'labels', 'class', 'category'] if x in fields), None)
            if namecol and labelcol:
                for row in reader:
                    name = Path(row[namecol].replace('\\', '/')).name
                    label = canonical(row[labelcol])
                    if label:
                        if name in csv_labels and csv_labels[name] != label:
                            raise ValueError(f'Conflicting CSV labels: {name}')
                        csv_labels[name] = label
    records, rejected = [], []
    for p in files:
        label, source = filename_info(p)
        supplied = csv_labels.get(p.name)
        try:
            if supplied and label and supplied != label:
                raise ValueError('CSV/name label conflict')
            label = supplied or label
            if label not in cfg['classes']:
                raise ValueError('Unknown class; add a documented alias')
            with Image.open(p) as im:
                im.load()
                w, h = im.size
            if min(w, h) < cfg['min_image_side']:
                raise ValueError('Image too small')
            sha = digest(p)
            records.append(dict(path=p.relative_to(root).as_posix(), label=label,
                                y=cfg['classes'].index(label), source=source, sha256=sha,
                                width=w, height=h, phash=f'{phash(p):016x}'))
        except Exception as e:
            rejected.append({'path': str(p.relative_to(root)), 'reason': str(e)})
    if not records:
        raise ValueError('No usable images')
    parent = list(range(len(records)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    tree = BKTree()
    by_sha = {}
    for i, row in enumerate(records):
        value = int(row['phash'], 16)
        for j in tree.query(value, cfg['phash_distance']):
            union(i, j)
        if row['sha256'] in by_sha:
            union(i, by_sha[row['sha256']])
        by_sha[row['sha256']] = i
        tree.add(value, i)
    groups = {}
    for i in range(len(records)):
        groups.setdefault(find(i), []).append(i)
    kept = []
    for indices in groups.values():
        group_id = min(records[i]['sha256'] for i in indices)
        conflict = len({records[i]['label'] for i in indices}) != 1
        for k, i in enumerate(indices):
            records[i]['group_id'] = group_id
            records[i]['decision'] = 'label_conflict' if conflict else ('keep' if k == 0 else 'near_duplicate')
            if records[i]['decision'] == 'keep':
                records[i]['image_id'] = records[i]['sha256']
                kept.append(records[i].copy())
    frame = pd.DataFrame(kept)
    if frame.empty:
        raise ValueError('All groups quarantined')
    rng = np.random.default_rng(cfg['split_seed'])
    split_names = ['train', 'val', 'calibration', 'test']
    frame['split'] = ''
    # Random split within true class; do not force fixed source-mixture counts
    # while claiming a pooled iid binomial certificate for that class.
    for label, part in frame.groupby('label'):
        n = len(part)
        if n < 8:
            raise ValueError(f'Too few images for class {label}: {n}')
        wanted = np.array([cfg['splits'][s] for s in split_names]) * n
        counts = np.floor(wanted).astype(int)
        for j in np.argsort(-(wanted-counts), kind='stable')[:n-counts.sum()]:
            counts[j] += 1
        if (counts == 0).any():
            raise ValueError('Empty stratum split')
        indices = rng.permutation(part.index)
        start = 0
        for s, count in zip(split_names, counts):
            frame.loc[indices[start:start+count], 'split'] = s
            start += count
    out.mkdir(parents=True, exist_ok=True)
    frame.sort_values('image_id').to_csv(out / 'manifest.csv', index=False)
    pd.DataFrame(records).to_csv(out / 'audit_all.csv', index=False)
    pd.DataFrame(rejected, columns=['path', 'reason']).to_csv(out / 'rejected.csv', index=False)
    write_json(out / 'audit.json', {'input_images': len(files), 'retained': len(frame),
               'counts': frame.groupby(['split', 'label']).size().unstack(fill_value=0).to_dict(),
               'decisions': dict(Counter(r['decision'] for r in records)),
               'source_status': 'filename proxy only; not independently verified clinical provenance',
               'manifest_sha256': digest(out / 'manifest.csv')})
    return frame


def validate_manifest(cfg, verify_bytes=False):
    path = resolve_path(cfg['output_dir']) / 'data/manifest.csv'
    if digest(path) != read_json(path.parent/'audit.json')['manifest_sha256']:
        raise ValueError('Manifest changed after data audit')
    frame = pd.read_csv(path, keep_default_na=False)
    if frame.empty or frame.image_id.duplicated().any() or frame.group_id.duplicated().any():
        raise ValueError('Manifest must retain exactly one image per duplicate group')
    if not set(frame.split).issubset({'train', 'val', 'calibration', 'test'}):
        raise ValueError('Unknown manifest split')
    if not all(row.label == cfg['classes'][row.y] and row.image_id == row.sha256
               for row in frame.itertuples() if row.y in range(4)) or not frame.y.isin(range(4)).all():
        raise ValueError('Manifest class order or image identity mismatch')
    for s in ['train', 'val', 'calibration', 'test']:
        if set(frame.loc[frame.split == s, 'y']) != {0, 1, 2, 3}:
            raise ValueError(f'Missing class in {s}')
    root = resolve_path(cfg['data_dir'])
    for row in frame.itertuples():
        p = (root / row.path).resolve()
        if not p.is_relative_to(root.resolve()) or not p.is_file():
            raise ValueError(f'Invalid image path: {row.path}')
        if verify_bytes and digest(p) != row.sha256:
            raise ValueError(f'Image changed after audit: {row.path}')
    return frame
