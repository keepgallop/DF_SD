import torch.nn as nn
from utils.utils import weights_init
from torch.autograd import Variable
import torch


class ResBlock(nn.Module):

    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [
            Conv2dBlock(dim,
                        dim,
                        3,
                        1,
                        1,
                        norm=norm,
                        activation=activation,
                        pad_type=pad_type)
        ]
        model += [
            Conv2dBlock(dim,
                        dim,
                        3,
                        1,
                        1,
                        norm=norm,
                        activation='none',
                        pad_type=pad_type)
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class LayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class AdaptiveInstanceNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(x_reshaped, running_mean, running_var, self.weight,
                           self.bias, True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class ResBlocks(nn.Module):

    def __init__(self,
                 num_blocks,
                 dim,
                 norm='in',
                 activation='relu',
                 pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [
                ResBlock(dim,
                         norm=norm,
                         activation=activation,
                         pad_type=pad_type)
            ]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(
                torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class Conv2dBlock(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 kernel_size,
                 stride,
                 padding=0,
                 norm='none',
                 activation='relu',
                 pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(
                nn.Conv2d(input_dim,
                          output_dim,
                          kernel_size,
                          stride,
                          bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim,
                                  output_dim,
                                  kernel_size,
                                  stride,
                                  bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ContentEncoder(nn.Module):

    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ,
                 pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [
            Conv2dBlock(input_dim,
                        dim,
                        7,
                        1,
                        3,
                        norm=norm,
                        activation=activ,
                        pad_type=pad_type)
        ]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [
                Conv2dBlock(dim,
                            2 * dim,
                            4,
                            2,
                            1,
                            norm=norm,
                            activation=activ,
                            pad_type=pad_type)
            ]
            dim *= 2
        # residual blocks
        self.model += [
            ResBlocks(n_res,
                      dim,
                      norm=norm,
                      activation=activ,
                      pad_type=pad_type)
        ]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):

    def __init__(self,
                 n_upsample,
                 n_res,
                 dim,
                 output_dim,
                 res_norm='adain',
                 activ='relu',
                 pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [
            ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)
        ]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [
                nn.Upsample(scale_factor=2),
                Conv2dBlock(dim,
                            dim // 2,
                            5,
                            1,
                            2,
                            norm='ln',
                            activation=activ,
                            pad_type=pad_type)
            ]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [
            Conv2dBlock(dim,
                        output_dim,
                        7,
                        1,
                        3,
                        norm='none',
                        activation='tanh',
                        pad_type=pad_type)
        ]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class VAE(nn.Module):
    # VAE architecture
    def __init__(self,
                 input_dim,
                 dim=100,
                 n_downsample=2,
                 n_res=4,
                 activ='relu',
                 pad_type='reflect'):
        super(VAE, self).__init__()
        # content encoder
        self.enc = ContentEncoder(n_downsample,
                                  n_res,
                                  input_dim,
                                  dim,
                                  'in',
                                  activ,
                                  pad_type=pad_type)
        self.dec = Decoder(n_downsample + 1,
                           n_res,
                           self.enc.output_dim,
                           input_dim,
                           res_norm='in',
                           activ=activ,
                           pad_type=pad_type)
        weights_init(self.enc)
        weights_init(self.dec)

    def forward(self, images):
        # This is a reduced VAE implementation where we assume the outputs are multivariate Gaussian distribution with mean = hiddens and std_dev = all ones.
        hiddens, _ = self.encode(images)
        if self.training == True:
            noise = Variable(
                torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))

            images_recon = self.decode(hiddens + noise)
        else:
            images_recon = self.decode(hiddens)
        return images_recon

    def encode(self, images):
        hiddens = self.enc(images)
        noise = Variable(
            torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
        return hiddens, noise

    def decode(self, hiddens):
        images = self.dec(hiddens)
        return images
