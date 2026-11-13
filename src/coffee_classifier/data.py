"""RGB preprocessing and deterministic, exact-duplicate-disjoint splits."""
import hashlib
import random
from pathlib import Path
import numpy as np
from PIL import Image

CLASSES = ['Dark', 'Green', 'Light', 'Medium']
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def image_array(path, size=50):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB').resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def fingerprint(path):
    with Image.open(path) as image:
        image = image.convert('RGB')
        return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def prepare(root, seed=42, val_fraction=0.1):
    root = Path(root)
    if not 0 < val_fraction < 1:
        raise ValueError('val_fraction must be between zero and one')
    rows, labels = [], {}
    for split in ['test', 'train']:
        for label, name in enumerate(CLASSES):
            directory = root / split / name
            paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in EXTENSIONS)
            if not paths:
                raise ValueError(f'No images in {directory}')
            for path in paths:
                digest = fingerprint(path)  # corrupt images fail visibly
                if digest in labels and labels[digest] != label:
                    raise ValueError(f'Conflicting labels for identical image: {path}')
                labels[digest] = label
                rows.append(dict(path=path.relative_to(root).as_posix(), label=label, source_split=split, sha256=digest))
    seen, kept, excluded = set(), [], []
    # Hold out the supplied test images first; retain only one copy of every image.
    for row in rows:
        if row['sha256'] in seen:
            excluded.append(row)
        else:
            seen.add(row['sha256'])
            kept.append(dict(row, split=row['source_split']))
    rng = random.Random(seed)
    for label in range(len(CLASSES)):
        group = [r for r in kept if r['split'] == 'train' and r['label'] == label]
        if len(group) < 2:
            raise ValueError(f'Insufficient unique training images for {CLASSES[label]}')
        rng.shuffle(group)
        for row in group[:max(1, min(len(group)-1, round(len(group)*val_fraction)))]:
            row['split'] = 'val'
    return dict(classes=CLASSES, seed=seed, val_fraction=val_fraction, hash_method='sha256 of native-size decoded RGB pixels and dimensions', rows=kept, excluded=excluded,
                counts={s: sum(r['split']==s for r in kept) for s in ['train','val','test']}, original_count=len(rows))


def load_split(root, manifest, split, size):
    rows = [r for r in manifest['rows'] if r['split']==split]
    return np.stack([image_array(Path(root)/r['path'], size) for r in rows]), np.array([r['label'] for r in rows])
