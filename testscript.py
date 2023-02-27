import torch
from CA import CA2d, CATranspose2d

x = torch.randn(5, 3, 20, 24)
m = CA2d(3, 10, stride=2)
mt = CATranspose2d(10, 15, stride=2)
mt(m(x))
