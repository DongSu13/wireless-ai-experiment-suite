"""Small CPU experiments on synthetic data; outputs are never overwritten."""

import argparse
import json
from pathlib import Path

import numpy as np


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=('csi', 'beam', 'csi-numpy', 'beam-numpy',
                                     'quantized', 'sparse'), default='csi-numpy')
    p.add_argument('--arch', default=None, help='CSI or beam architecture registry key')
    p.add_argument('--latent-dim', type=positive_int, default=16)
    p.add_argument('--feedback-bits', type=positive_int, default=128)
    p.add_argument('--train-samples', type=positive_int, default=32)
    p.add_argument('--val-samples', type=positive_int, default=8)
    p.add_argument('--test-samples', type=positive_int, default=8)
    p.add_argument('--epochs', type=positive_int, default=2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, help='optional NEW JSON file, outside the repository')
    return p


def make_splits(n_train, n_val, n_test, seed, *, square=False):
    from .channel import generate_hcsi_down_all
    children = np.random.SeedSequence(seed).spawn(3)
    return tuple(generate_hcsi_down_all(n, seed=int(s.generate_state(1)[0]),
                                       n_subbands=32 if square else 12)
                 for n, s in zip((n_train, n_val, n_test), children))


def two_channel(h):
    return np.stack((h.real, h.imag), axis=1).astype(np.float32)


def _numpy_experiment(args, splits):
    from .numpy_csi import CSIFCAE, Adam
    from .numpy_beam import BeamformerMLP
    beam = args.task == 'beam-numpy'
    arrays = tuple(two_channel(h[:, 0] if beam else h).reshape(len(h), -1)
                   for h in splits)
    targets = tuple(x / np.linalg.norm(x, axis=1, keepdims=True) if beam else x
                    for x in arrays)
    model = (BeamformerMLP(arrays[0].shape[1], 64, seed=args.seed) if beam else
             CSIFCAE(arrays[0].shape[1], 64, args.latent_dim, seed=args.seed))
    optimizer = Adam(model.params_dict(), lr=1e-3)
    best, best_params = float('inf'), None
    for _ in range(args.epochs):
        predicted, cache = model.forward(arrays[0])
        grads = model.backward(cache, 2 * (predicted - targets[0]) / predicted.size)
        optimizer.step(grads)
        val, _ = model.forward(arrays[1])
        loss = float(np.mean((val - targets[1]) ** 2))
        if loss < best:
            best, best_params = loss, {k: v.copy() for k, v in model.params_dict().items()}
    for k, v in model.params_dict().items():
        v[:] = best_params[k]
    predicted, _ = model.forward(arrays[2])
    shape = (len(predicted), 2, 32) if beam else (len(predicted), 2, 12, 32)
    predicted = predicted.reshape(shape)
    return predicted[:, 0] + 1j * predicted[:, 1], best


def _torch_experiment(args, splits):
    import torch
    from .csi_models import build_csi_model
    from .beam_models import build_beam_model
    from .quantized_csi import QuantizedCsiAutoencoder
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    beam = args.task == 'beam'
    inputs = tuple(torch.from_numpy(two_channel(h[:, 0] if beam else h)) for h in splits)
    targets = tuple(x / x.square().sum((1, 2), keepdim=True).sqrt() if beam else x
                    for x in inputs)
    if args.task == 'quantized':
        model = QuantizedCsiAutoencoder(args.feedback_bits)
    elif beam:
        model = build_beam_model(args.arch or 'mlp')
    else:
        model = build_csi_model(args.arch or 'cnn', latent_dim=args.latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def predict(x):
        output = model(x)
        return output[0] if isinstance(output, tuple) else output

    best, best_state = float('inf'), None
    for _ in range(args.epochs):
        model.train()
        loss = (predict(inputs[0]) - targets[0]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float((predict(inputs[1]) - targets[1]).square().mean())
        if val_loss < best:
            best, best_state = val_loss, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        output = predict(inputs[2]).numpy()
    return output[:, 0] + 1j * output[:, 1], best


def run(args):
    from .validation.metrics import (nmse_per_sample, nmse_db_from_samples,
                                    beam_gain_per_sample, complex_sgcs_per_sample)
    if args.seed < 0 or args.seed >= 2 ** 32:
        raise ValueError('seed must be in [0, 2**32)')
    if args.arch and args.task not in ('csi', 'beam'):
        raise ValueError('--arch is supported only for csi and beam')
    if args.task in ('quantized', 'sparse') and args.feedback_bits % (8 if args.task == 'sparse' else 4):
        raise ValueError('feedback budget must be divisible by 8 (sparse) or 4 (quantized)')
    square = args.task in ('quantized', 'sparse')
    splits = make_splits(args.train_samples, args.val_samples, args.test_samples,
                         args.seed, square=square)
    result = {'task': args.task, 'seed': args.seed, 'train_samples': args.train_samples,
              'validation_samples': args.val_samples, 'test_samples': args.test_samples,
              'epochs': args.epochs, 'evaluation_split': 'test',
              'data': 'synthetic correlated multipath frequency-domain channel'}
    if args.task == 'sparse':
        from .sparse_feedback import fit_fixed_sparse, reconstruct
        model = fit_fixed_sparse(splits[0], max_coefficients=args.feedback_bits // 8)
        prediction, used = reconstruct(splits[2], model, args.feedback_bits)
        result.update(feedback_bits=used, side_information='train-fitted mean, indices and scales shared offline')
    elif args.task.endswith('-numpy'):
        prediction, best = _numpy_experiment(args, splits)
        result['best_validation_mse'] = best
    else:
        prediction, best = _torch_experiment(args, splits)
        result['best_validation_mse'] = best
    if args.task.startswith('beam'):
        result['mean_normalized_beam_gain'] = float(beam_gain_per_sample(splits[2][:, 0], prediction).mean())
    else:
        errors = nmse_per_sample(splits[2], prediction)
        result.update(nmse_linear_mean=float(errors.mean()), nmse_db=nmse_db_from_samples(errors))
        result['mean_frequency_domain_sgcs'] = float(complex_sgcs_per_sample(splits[2], prediction).mean())
        if args.task == 'quantized':
            result['feedback_bits'] = args.feedback_bits
        elif args.task != 'sparse':
            result['latent_dim'] = args.latent_dim
    if args.task in ('csi', 'beam'):
        result['architecture'] = args.arch or ('cnn' if args.task == 'csi' else 'mlp')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.out and args.out.exists():
        raise FileExistsError('choose a new output file; existing files are preserved')
    result = run(args)
    serialized = json.dumps(result, indent=2, allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x', encoding='utf-8') as stream:
            stream.write(serialized + '\n')
    print(serialized)
    return result
