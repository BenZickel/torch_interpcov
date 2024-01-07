# torch_interpcov

Covariance matrix interpolation in PyTorch implemented as a torch `torch.nn.Module` called `InterpolateCovariance` with the following features:
- Interpolated covariance matrix maintains positive definiteness.
- Convergence to the original covariance matrix near the input coordinates.

## Tests

```
python -m torch_interpcov
```
