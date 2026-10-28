"""TensorFlow is imported lazily so audit/statistics work without it."""
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from scipy.special import softmax

from .common import digest, read_json, resolve_path, write_json


def tensorflow():
    try:
        import tensorflow as tf
    except ImportError as e:
        raise RuntimeError('Install requirements.txt (CPU) or requirements-wsl-gpu.txt (Linux/WSL2 GPU).') from e
    return tf


def read_image(path, size):
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert('RGB').resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)


def dataset(frame, cfg, training, seed):
    tf = tensorflow()
    root = resolve_path(cfg['data_dir'])
    paths, labels = frame.path.tolist(), frame.y.to_numpy(dtype=np.int32)
    size = cfg['image_size']
    def generate():
        for path, label in zip(paths, labels):
            yield read_image(root/path, size), label
    ds = tf.data.Dataset.from_generator(generate, output_signature=(
        tf.TensorSpec((size, size, 3), tf.float32), tf.TensorSpec((), tf.int32)))
    if training:
        ds = ds.shuffle(min(len(frame), 4096), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(cfg['batch_size'])
    if training:
        aug = tf.keras.Sequential([tf.keras.layers.RandomFlip('horizontal', seed=seed),
            tf.keras.layers.RandomRotation(0.06, seed=seed+1),
            tf.keras.layers.RandomZoom(0.1, seed=seed+2),
            tf.keras.layers.RandomContrast(0.15, seed=seed+3)])
        ds = ds.map(lambda x, y: (tf.clip_by_value(aug(x, training=True), 0, 255), y), num_parallel_calls=1)
    return ds.prefetch(1)


def build_model(name, cfg):
    tf = tensorflow()
    inputs = tf.keras.Input((cfg['image_size'], cfg['image_size'], 3), name='rgb_0_255')
    weights = cfg.get('weights')
    if weights in {'none', 'null', ''}:
        weights = None
    if weights not in {None, 'imagenet'}:
        weights = str(resolve_path(weights))
    if name == 'mobilenetv2':
        x = tf.keras.layers.Rescaling(1/127.5, offset=-1)(inputs)
        backbone = tf.keras.applications.MobileNetV2(include_top=False, weights=weights,
                    input_shape=(cfg['image_size'], cfg['image_size'], 3))
    elif name == 'efficientnetb0':
        # EfficientNetB0 has its own input rescaling; it is NOT EfficientNet-Lite0.
        x = inputs
        backbone = tf.keras.applications.EfficientNetB0(include_top=False, weights=weights,
                    input_shape=(cfg['image_size'], cfg['image_size'], 3))
    elif name == 'tinycnn':
        x = tf.keras.layers.Rescaling(1/255)(inputs)
        for width in [16, 32, 64]:
            x = tf.keras.layers.Conv2D(width, 3, strides=2, padding='same', activation='relu')(x)
        backbone = None
    else:
        raise ValueError(name)
    if backbone is not None:
        backbone.trainable = False
        x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(4, name='logits')(x)
    return tf.keras.Model(inputs, outputs), backbone


def train(frame, cfg, name, seed, out):
    tf = tensorflow()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    tf.config.experimental.enable_op_determinism()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'training_complete.json').exists():
        info = read_json(out/'training_complete.json')
        if not (out/'best.keras').is_file() or digest(out/'best.keras') != info['model_sha256']:
            raise ValueError('Saved model changed or missing')
        return
    model, backbone = build_model(name, cfg)
    tr = frame[frame.split == 'train']
    va = frame[frame.split == 'val']
    train_ds, val_ds = dataset(tr, cfg, True, seed), dataset(va, cfg, False, seed)
    counts = tr.y.value_counts()
    weights = {int(c): float(len(tr)/(4*n)) for c, n in counts.items()} if cfg['class_weight'] else None
    # Global best spans both stages; no calibration/test images used here.
    from sklearn.metrics import f1_score
    class ValidationCheckpoint(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best, self.wait, self.rows, self.stage = -1.0, 0, [], 'head'
        def on_epoch_end(self, epoch, logs=None):
            logits = self.model.predict(val_ds, verbose=0)
            score = float(f1_score(va.y.to_numpy(), logits.argmax(axis=1), labels=range(4), average='macro', zero_division=0))
            self.rows.append({'stage': self.stage, 'epoch': epoch+1, 'val_macro_f1': score,
                              **{k: float(v) for k, v in (logs or {}).items()}})
            print(f'[{self.stage}] epoch={epoch+1} val_macro_f1={score:.5f}', flush=True)
            if score > self.best:
                self.best, self.wait = score, 0
                self.model.save(out/'best.keras')
            else:
                self.wait += 1
                if self.wait >= cfg['patience']:
                    self.model.stop_training = True
            write_json(out/'history.json', self.rows)
    callback = ValidationCheckpoint()
    def compile_model(lr):
        model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    compile_model(cfg['learning_rate_head'])
    model.fit(train_ds, epochs=cfg['head_epochs'], callbacks=[callback], class_weight=weights, verbose=2)
    if backbone is not None and cfg['finetune_epochs'] > 0:
        # Restore the best head weights before fine-tuning.
        model.load_weights(out/'best.keras')
        backbone.trainable = True
        cutoff = max(0, len(backbone.layers)-cfg['unfreeze_last'])
        for i, layer in enumerate(backbone.layers):
            layer.trainable = i >= cutoff and not isinstance(layer, tf.keras.layers.BatchNormalization)
        callback.stage, callback.wait = 'finetune', 0
        compile_model(cfg['learning_rate_finetune'])
        model.fit(train_ds, epochs=cfg['finetune_epochs'], callbacks=[callback], class_weight=weights, verbose=2)
    write_json(out/'training_complete.json', {'best_validation_macro_f1': callback.best,
               'model_sha256': digest(out/'best.keras'), 'seed': seed, 'backbone': name,
               'tensorflow_version': tf.__version__})


def export(frame, cfg, out):
    tf = tensorflow()
    info = read_json(out/'training_complete.json')
    if digest(out/'best.keras') != info['model_sha256']:
        raise ValueError('Training checkpoint changed after completion')
    model = tf.keras.models.load_model(out/'best.keras', compile=False)
    saved = out/'saved_model'
    if not saved.exists():
        model.export(str(saved), format='tf_saved_model')
    train_rows = frame[frame.split == 'train'].sample(frac=1, random_state=cfg['split_seed'])
    representatives = train_rows.head(cfg['representative_images'])
    write_json(out/'representative_ids.json', representatives.image_id.tolist())
    root = resolve_path(cfg['data_dir'])
    for fmt in cfg['formats']:
        target = out/f'model_{fmt}.tflite'
        marker = out/f'export_{fmt}.json'
        if marker.exists():
            if digest(target) != read_json(marker)['sha256']:
                raise ValueError('TFLite file hash mismatch')
            continue
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved))
        if fmt == 'int8':
            def representative():
                for r in representatives.itertuples():
                    yield [read_image(root/r.path, cfg['image_size'])[None, ...]]
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        blob = converter.convert()
        target.write_bytes(blob)
        interpreter = tf.lite.Interpreter(model_path=str(target), num_threads=cfg['interpreter_threads'])
        interpreter.allocate_tensors()
        inp, op = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
        if fmt == 'int8':
            if inp['dtype'] != np.int8 or op['dtype'] != np.int8:
                raise ValueError('INT8 I/O verification failed')
            float_tensors = [x['name'] for x in interpreter.get_tensor_details() if np.issubdtype(x['dtype'], np.floating)]
            if float_tensors:
                raise ValueError(f'Unexpected floating tensors in full INT8 graph: {float_tensors}')
        write_json(marker, {'sha256': digest(target), 'size_bytes': len(blob),
            'input_dtype': str(inp['dtype']), 'output_dtype': str(op['dtype']),
            'input_quantization': list(inp['quantization']), 'output_quantization': list(op['quantization']),
            'representative_split': 'train', 'representative_count': len(representatives)})


def infer(frame, cfg, out, fmt, split):
    if split == 'calibration':
        from .pipeline import verify_selection_freeze
        verify_selection_freeze(out)
    if split == 'test':
        from .pipeline import verify_policy_freeze
        verify_policy_freeze(out)
    tf = tensorflow()
    target = out/f'predictions_{fmt}_{split}.npz'
    if digest(out/f'model_{fmt}.tflite') != read_json(out/f'export_{fmt}.json')['sha256']:
        raise ValueError('TFLite model changed after export')
    part = frame[frame.split == split].sort_values('image_id')
    if target.exists():
        return target
    interpreter = tf.lite.Interpreter(model_path=str(out/f'model_{fmt}.tflite'), num_threads=cfg['interpreter_threads'])
    interpreter.allocate_tensors()
    inp, op = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    root = resolve_path(cfg['data_dir'])
    logits, timings = [], []
    for i, row in enumerate(part.itertuples()):
        x = read_image(root/row.path, cfg['image_size'])[None, ...]
        if np.issubdtype(inp['dtype'], np.integer):
            scale, zero = inp['quantization']
            if scale <= 0:
                raise ValueError('Invalid input quantization scale')
            limits = np.iinfo(inp['dtype'])
            x = np.clip(np.rint(x/scale+zero), limits.min, limits.max).astype(inp['dtype'])
        else:
            x = x.astype(inp['dtype'])
        interpreter.set_tensor(inp['index'], x)
        if i == 0:
            for _ in range(10):
                interpreter.invoke()
        begin = time.perf_counter()
        interpreter.invoke()
        timings.append((time.perf_counter()-begin)*1000)
        raw = interpreter.get_tensor(op['index'])[0]
        if np.issubdtype(op['dtype'], np.integer):
            scale, zero = op['quantization']
            raw = (raw.astype(np.float32)-zero)*scale
        logits.append(raw)
    logits = np.asarray(logits)
    tmp = target.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, image_id=part.image_id.to_numpy(dtype=str), y=part.y.to_numpy(dtype=np.int64),
            p=softmax(logits, axis=1), logits=logits, source=part.source.to_numpy(dtype=str))
    tmp.replace(target)
    write_json(out/f'latency_{fmt}_{split}.json', {'median_invoke_ms': float(np.median(timings)),
              'p95_invoke_ms': float(np.quantile(timings, 0.95)), 'warmup_invocations': 10,
              'threads': cfg['interpreter_threads'], 'scope': 'local host CPU invoke only; excludes loading/preprocessing; not smartphone latency'})
    return target
