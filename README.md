# torch_interpcov

Covariance matrix interpolation in PyTorch implemented as a torch `torch.nn.Module` called `InterpolateCovariance` with the following features:
- Interpolated covariance matrix maintains positive definiteness.
- Convergence to the original covariance matrix near the input coordinates.

## Use case

The main use case is extending a multivariate gaussian distribution $`Z(x_i)`$ defined at $`N`$ descrete strictly increasing series of points $`x_1,...,x_N`$ to the continuous range $`[x_1,x_N]`$. Given the covariance of $`Z(x_i)`$ defined at $`x_1,...,x_N`$ the `InterpolateCovariance` module outputs the covariance at an arbitrary set of points.

## Tests

```
python -m torch_interpcov
```
