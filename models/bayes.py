import torch
from torch import nn
from torch.nn import functional as F

class BayesianConv2d(nn.Module):
    # Implements Bayesian Conv2d layer, by drawing them using Weight Uncertanity on Neural Networks algorithm
    """
    Bayesian Linear layer, implements a Convolution 2D layer.

    Its objective is be interactable with torch nn.Module API, being able even to be chained in nn.Sequential models with other non-this-lib layers

    parameters:
        in_channels: int -> incoming channels for the layer
        out_channels: int -> output channels for the layer
        kernel_size : tuple (int, int) -> size of the kernels for this convolution layer
        groups : int -> number of groups on which the convolutions will happend
        padding : int -> size of padding (0 if no padding)
        dilation int -> dilation of the weights applied on the input tensor


        bias: bool -> whether the bias will exist (True) or set to zero (False)
        prior_sigma_1: float -> prior sigma on the mixture prior distribution 1
        prior_sigma_2: float -> prior sigma on the mixture prior distribution 2
        prior_pi: float -> pi on the scaled mixture prior
        posterior_mu_init float -> posterior mean for the weight mu init
        posterior_rho_init float -> posterior mean for the weight rho init
        freeze: bool -> wheter the model will start with frozen(deterministic) weights, or not

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 groups=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 posterior_mu_init=0,
                 posterior_rho_init=-6.0,
                 freeze=False,
                 prior_dist=None):
        super().__init__()

        # our main parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.freeze = freeze
        self.kernel_size = kernel_size
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        self.prior_dist = prior_dist
        self.reg_square_mu = 0
        self.reg_square_sigma = 0
        self.reg_log_sigma = 0

        # our weights
        self.weight_mu = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *kernel_size).normal_(posterior_mu_init, 0.1))
        self.weight_rho = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *kernel_size).normal_(posterior_rho_init, 0.1))
        self.weight_sampler = TrainableRandomDistribution(self.weight_mu, self.weight_rho)

        # our biases
        if self.bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_channels).normal_(posterior_mu_init, 0.1))
            self.bias_rho = nn.Parameter(torch.Tensor(out_channels).normal_(posterior_rho_init, 0.1))
            self.bias_sampler = TrainableRandomDistribution(self.bias_mu, self.bias_rho)

        else:
            self.register_buffer('bias_zero', torch.zeros(self.out_channels))



    def forward(self, x):
        # Forward with uncertain weights, fills bias with zeros if layer has no bias
        # Also calculates the complexity cost for this sampling
        if self.freeze:
            return self.forward_frozen(x)

        w = self.weight_sampler.sample()

        if self.bias:
            b = self.bias_sampler.sample()

        else:
            b = self.bias_zero

        self.reg_square_mu = torch.sum(torch.square(self.weight_mu))
        self.reg_square_sigma = torch.sum(torch.square(self.weight_sampler.sigma))
        self.reg_log_sigma = torch.sum(torch.log(self.weight_sampler.sigma))

        return F.conv2d(input=x,
                        weight=w,
                        bias=b,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups)

    def forward_frozen(self, x):
        # Computes the feedforward operation with the expected value for weight and biases (frozen-like)

        if self.bias:
            bias = self.bias_mu
            assert bias is self.bias_mu, "The bias inputed should be this layer parameter, not a clone."
        else:
            bias = self.bias_zero

        return F.conv2d(input=x,
                        weight=self.weight_mu,
                        bias=bias,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups)


class TrainableRandomDistribution(nn.Module):
    # Samples weights for variational inference as in Weights Uncertainity on Neural Networks (Bayes by backprop paper)
    # Calculates the variational posterior part of the complexity part of the loss
    def __init__(self, mu, rho):
        super().__init__()

        self.mu = nn.Parameter(mu)
        self.rho = nn.Parameter(rho)
        self.register_buffer('eps_w', torch.Tensor(self.mu.shape))
        self.sigma = None
        self.w = None

    def sample(self):
        """
        Samples weights by sampling form a Normal distribution, multiplying by a sigma, which is
        a function from a trainable parameter, and adding a mean

        sets those weights as the current ones

        returns:
            torch.tensor with same shape as self.mu and self.rho
        """

        self.eps_w.data.normal_()
        self.sigma = torch.log1p(torch.exp(self.rho))
        self.w = self.mu + self.sigma * self.eps_w
        return self.w


def regularisation_error(model, lambda_1, lambda_2):
    reg_error = 0
    for module in model.modules():
        if isinstance(module, BayesianConv2d):
            reg_error += (module.reg_square_mu + module.reg_square_sigma) * lambda_1 - module.reg_log_sigma * lambda_2
    return reg_error

