# ###############################################################################
# # Part of code from
# # https://github.com/NVlabs/MUNIT/blob/master/networks.py
# ###############################################################################
# import sys

# sys.path.append('../')
# import torch.nn as nn
# from utils.utils import weights_init
# import torch

# class AE(nn.Module):

#     def __init__(self,
#                  input_dim,
#                  output_dim,
#                  dim,
#                  n_blk,
#                  norm='none',
#                  activ='relu'):
#         super(AE, self).__init__()
#         self.model = []
#         self.model += [
#             LinearBlock(input_dim, dim, norm=norm, activation=activ)
#         ]
#         for i in range(n_blk - 2):
#             self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
#         self.model += [
#             LinearBlock(dim, output_dim, norm='none', activation='tanh')
#         ]
#         self.model = nn.Sequential(*self.model)
#         weights_init(self.model)

#     def forward(self, x):
#         sz = x.shape
#         return self.model(x.view(x.size(0), -1)).view(sz)

# class LinearBlock(nn.Module):

#     def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
#         super(LinearBlock, self).__init__()
#         use_bias = True
#         # initialize fully connected layer
#         if norm == 'sn':
#             self.fc = SpectralNorm(
#                 nn.Linear(input_dim, output_dim, bias=use_bias))
#         else:
#             self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

#         # initialize normalization
#         norm_dim = output_dim
#         if norm == 'bn':
#             self.norm = nn.BatchNorm1d(norm_dim)
#         elif norm == 'in':
#             self.norm = nn.InstanceNorm1d(norm_dim)
#         elif norm == 'ln':
#             self.norm = LayerNorm(norm_dim)
#         elif norm == 'none' or norm == 'sn':
#             self.norm = None
#         else:
#             assert 0, "Unsupported normalization: {}".format(norm)

#         # initialize activation
#         if activation == 'relu':
#             self.activation = nn.ReLU(inplace=True)
#         elif activation == 'lrelu':
#             self.activation = nn.LeakyReLU(0.2, inplace=True)
#         elif activation == 'prelu':
#             self.activation = nn.PReLU()
#         elif activation == 'selu':
#             self.activation = nn.SELU(inplace=True)
#         elif activation == 'tanh':
#             self.activation = nn.Tanh()
#         elif activation == 'none':
#             self.activation = None
#         else:
#             assert 0, "Unsupported activation: {}".format(activation)

#     def forward(self, x):
#         out = self.fc(x)
#         if self.norm:
#             out = self.norm(out)
#         if self.activation:
#             out = self.activation(out)
#         return out

# class LayerNorm(nn.Module):

#     def __init__(self, num_features, eps=1e-5, affine=True):
#         super(LayerNorm, self).__init__()
#         self.num_features = num_features
#         self.affine = affine
#         self.eps = eps

#         if self.affine:
#             self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
#             self.beta = nn.Parameter(torch.zeros(num_features))

#     def forward(self, x):
#         shape = [-1] + [1] * (x.dim() - 1)
#         # print(x.size())
#         if x.size(0) == 1:
#             # These two lines run much faster in pytorch 0.4 than the two lines listed below.
#             mean = x.view(-1).mean().view(*shape)
#             std = x.view(-1).std().view(*shape)
#         else:
#             mean = x.view(x.size(0), -1).mean(1).view(*shape)
#             std = x.view(x.size(0), -1).std(1).view(*shape)

#         x = (x - mean) / (std + self.eps)

#         if self.affine:
#             shape = [1, -1] + [1] * (x.dim() - 2)
#             x = x * self.gamma.view(*shape) + self.beta.view(*shape)
#         return x

# def l2normalize(v, eps=1e-12):
#     return v / (v.norm() + eps)

# class SpectralNorm(nn.Module):
#     """
#     Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
#     and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
#     """

#     def __init__(self, module, name='weight', power_iterations=1):
#         super(SpectralNorm, self).__init__()
#         self.module = module
#         self.name = name
#         self.power_iterations = power_iterations
#         if not self._made_params():
#             self._make_params()

#     def _update_u_v(self):
#         u = getattr(self.module, self.name + "_u")
#         v = getattr(self.module, self.name + "_v")
#         w = getattr(self.module, self.name + "_bar")

#         height = w.data.shape[0]
#         for _ in range(self.power_iterations):
#             v.data = l2normalize(
#                 torch.mv(torch.t(w.view(height, -1).data), u.data))
#             u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

#         # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
#         sigma = u.dot(w.view(height, -1).mv(v))
#         setattr(self.module, self.name, w / sigma.expand_as(w))

#     def _made_params(self):
#         try:
#             u = getattr(self.module, self.name + "_u")
#             v = getattr(self.module, self.name + "_v")
#             w = getattr(self.module, self.name + "_bar")
#             return True
#         except AttributeError:
#             return False

#     def _make_params(self):
#         w = getattr(self.module, self.name)

#         height = w.data.shape[0]
#         width = w.view(height, -1).data.shape[1]

#         u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
#         v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
#         u.data = l2normalize(u.data)
#         v.data = l2normalize(v.data)
#         w_bar = nn.Parameter(w.data)

#         del self.module._parameters[self.name]

#         self.module.register_parameter(self.name + "_u", u)
#         self.module.register_parameter(self.name + "_v", v)
#         self.module.register_parameter(self.name + "_bar", w_bar)

#     def forward(self, *args):
#         self._update_u_v()
#         return self.module.forward(*args)

# build autoencoder network

import torch.nn as nn
import torch.nn.functional as F


class AE(nn.Module):

    def __init__(self):
        super(AE, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 8, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 8, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(8, 3, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(3)
        self.convt1 = nn.ConvTranspose2d(8, 8, 2, stride=2)
        self.convt2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.convt3 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        self.convt4 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.pool = nn.MaxPool2d(2, 2)
        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight.data)
                nn.init.normal(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(self.bn1(x)))  # out [16, N/2, N/2, 1]
        x = self.conv2(x)
        x = self.pool(F.relu(self.bn2(x)))  # out [8, N/4, N/4, 1]
        x = self.conv3(x)
        x = self.pool(F.relu(self.bn2(x)))  # out [8, N/8, N/8, 1]

        x = self.convt1(x)
        x = F.relu(self.bn2(x))  # out [8, N/4, N/4, 1]
        x = self.convt2(x)
        x = F.relu(self.bn1(x))  # out [16, N/2, N/2, 1]
        x = self.convt3(x)
        x = F.relu(self.bn3(x))  # out [32, N, N, 1]
        x = self.convt4(x)
        x = F.relu(self.bn1(x))  # out [16, 2N, 2N, 1]

        x = self.conv2(x)
        x = F.relu(self.bn2(x))  # out [8, 2N, 2N, 1]
        x = self.conv4(x)
        x = F.relu(self.bn4(x))  # out [3, 2N, 2N, 1]

        return x