from unittest import TestCase, main

import torch
from torch import nn

from pytorch_fused_lamb import Lamb


class SmallOptimizationTest(TestCase):
    """ Test LAMB optimizer on a simple optimization problem. """
    def test_fixed_optimization(self):
        y_hat = torch.randn(5, 4, requires_grad=True)
        y = torch.rand(5, 4)
        criterion = nn.MSELoss()
        optimizer = Lamb(params=(y_hat,), lr=1e-3)
        # Stop before full converge for faster runtime
        for _ in range(3000):
            optimizer.zero_grad()
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
        torch.testing.assert_close(y_hat, y, atol=1e-2, rtol=1e-2)

if __name__ == '__main__':
    main()
