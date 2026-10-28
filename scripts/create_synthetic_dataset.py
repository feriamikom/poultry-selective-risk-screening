"""Generate artificial RGB patterns solely for infrastructure tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from poultryrisk.common import ROOT

out = ROOT/'data/synthetic'
if out.exists() and list(out.glob('*.png')):
    raise SystemExit('Synthetic images already exist; reuse them.')
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(730)
names = ['healthy', 'cocci', 'salmo', 'ncd']
tints = [[100, 10, 10], [10, 100, 10], [10, 10, 100], [70, 70, 10]]
for c, name in enumerate(names):
    for i in range(80):
        image = np.clip(rng.uniform(0, 130, size=(64, 64, 3))+tints[c], 0, 255).astype('uint8')
        prefix = 'pcr' if i < 40 else ''
        Image.fromarray(image).save(out/f'{prefix}{name}.{i}.png')
print(f'Created 320 SYNTHETIC images in {out}. Never report these as poultry results.')
