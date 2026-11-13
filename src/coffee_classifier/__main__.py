import argparse
import json
import platform
from pathlib import Path
from .data import prepare, load_split, image_array, CLASSES


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description='Coffee roast classifier')
    commands = parser.add_subparsers(dest='command', required=True)
    audit = commands.add_parser('audit')
    audit.add_argument('--data', default='data')
    audit.add_argument('--output', default='reports/data-audit.json')
    train = commands.add_parser('train')
    train.add_argument('--data', default='data')
    train.add_argument('--config', default='configs/baseline.json')
    train.add_argument('--output', default='runs/baseline')
    train.add_argument('--epochs', type=int)
    predict = commands.add_parser('predict')
    predict.add_argument('--run', required=True)
    predict.add_argument('--image', required=True)
    args = parser.parse_args()
    if args.command == 'audit':
        manifest = prepare(args.data)
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        write(args.output, manifest)
        print(json.dumps(manifest['counts']))
        return
    import numpy as np
    import tensorflow as tf
    if args.command == 'predict':
        run = Path(args.run)
        metadata = json.loads((run/'metadata.json').read_text())
        model = tf.keras.models.load_model(run/'best.keras')
        scores = model(image_array(args.image,metadata['config']['image_size'])[None], training=False).numpy()[0]
        print(json.dumps(dict(zip(metadata['classes'],map(float,scores))),indent=2))
        return
    from .model import build_model
    from sklearn.metrics import classification_report, confusion_matrix
    config = json.loads(Path(args.config).read_text())
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if config['epochs'] < 1:
        raise ValueError('epochs must be positive')
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)  # never overwrite a previous experiment
    tf.keras.utils.set_random_seed(config['seed'])
    tf.config.experimental.enable_op_determinism()
    manifest = prepare(args.data,config['seed'],config['val_fraction'])
    write(out/'manifest.json',manifest)
    write(out/'metadata.json',dict(config=config,classes=CLASSES,preprocessing='Pillow RGB, bilinear resize, float32 / 255',python=platform.python_version(),tensorflow=tf.__version__,numpy=np.__version__))
    splits = {s:load_split(args.data,manifest,s,config['image_size']) for s in ['train','val','test']}
    # Limit dataset worker threads for small CPU/CI environments.
    def dataset(split, shuffle=False):
        x,y = splits[split]
        ds = tf.data.Dataset.from_tensor_slices((x,y))
        if shuffle:
            ds = ds.shuffle(len(y),seed=config['seed'])
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        return ds.batch(config['batch_size']).with_options(options)
    model = build_model(config['image_size'],config['l2_strength'],config['learning_rate'],config['seed'])
    callbacks = [tf.keras.callbacks.ModelCheckpoint(out/'best.keras',save_best_only=True,monitor='val_loss'),tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(patience=5,factor=0.5,min_lr=1e-5)]
    history = model.fit(dataset('train',True),validation_data=dataset('val'),epochs=config['epochs'],callbacks=callbacks)
    write(out/'history.json',history.history)
    model = tf.keras.models.load_model(out/'best.keras')
    x,y = splits['test']
    probabilities = np.concatenate([model(x[i:i+config['batch_size']],training=False).numpy() for i in range(0,len(x),config['batch_size'])])
    predictions = probabilities.argmax(axis=1)
    write(out/'test_metrics.json',dict(accuracy=float(np.mean(predictions==y)),classification_report=classification_report(y,predictions,labels=list(range(4)),target_names=CLASSES,output_dict=True,zero_division=0),confusion_matrix=confusion_matrix(y,predictions,labels=list(range(4))).tolist(),note='Exact duplicates removed; near-duplicates and acquisition-group leakage remain untested.'))
    np.savez(out/'test_predictions.npz',labels=y,probabilities=probabilities)
    print(f'Artifacts saved in {out}')


if __name__ == '__main__':
    main()
