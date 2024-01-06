import torch as pt

class InterpolateCovariance(pt.nn.Module):
    '''
    Interpolate the covariance matrix of a one dimensional discretely defined Gaussian process.

    Args:
        x: Input covariance matrix coordinates.
        xi: Output covariance matrix coordinates.
        add_Zero: If set to ``True``, a zero variance point is added at the origin.
            Default: ``False``

    Examples:
        >>> import torch as pt
        >>> from InterpolateCovariance import InterpolateCovariance
        >>> x = pt.Tensor([1, 2, 3, 4])
        >>> xi = pt.Tensor([0.1, 1.2, 3.6])
        >>> interp_cov = InterpolateCovariance(x,xi, add_zero=True)
        >>> cov = pt.randn(4, 4)
        >>> cov = pt.matmul(cov, cov.t())
        >>> covi = interp_cov(cov)
        >>> print(cov.shape, covi.shape)
        torch.Size([4, 4]) torch.Size([3, 3])
    '''

    def __init__(self, x, xi, add_zero=False):
        if add_zero:
            x = pt.cat([pt.zeros(1).to(x), x])

        if pt.any(pt.diff(x) <= 0):
            raise UserWarning('x must be strictly increasing')

        if xi.max() > x.max() or xi.min() < x.min():
            raise UserWarning('xi out of bounds')
        
        # Calculate interpolation data point continuous index
        cont_idx = pt.stack([(pt.clamp( xi, x[c], x[c + 1]) - x[c]) / (x[c + 1] - x[c]) 
                                                        for c in range(len(x) - 1)]).sum(0)

        # Get integer index and fraction
        idx = pt.clamp(cont_idx.floor(), 0, len(x) - 2 ).type(pt.LongTensor)
        frac = cont_idx - idx.to(cont_idx)

        # Create covariance interpolation matrix
        if add_zero:
            interp_mat = pt.zeros(len(xi), len(x)-1)
            non_zero_idx = pt.where(idx > 0)[0]
            interp_mat[non_zero_idx, idx[non_zero_idx] - 1] = 1 - frac[non_zero_idx]
            interp_mat[range(len(idx)), idx] = frac
        else:
            interp_mat = pt.zeros(len(xi), len(x))
            interp_mat[range(len(idx)), idx] = 1 - frac
            interp_mat[range(len(idx)), idx + 1] = frac

        super().__init__()
        self.x, self.xi, self.interp_mat, self.add_zero = x, xi, interp_mat, add_zero
        
    def forward(self, cov):
        return pt.matmul(pt.matmul(self.interp_mat, cov), self.interp_mat.t())
   
if __name__ == "__main__":
    import doctest
    doctest.testmod()
