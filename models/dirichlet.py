import torch.nn
from torch import nn
import math
from torch.distributions.dirichlet import *
from torch.distributions.kl import kl_divergence
import numpy as np

CHANNEL_EXP_SUMS = [1,1,1]
CURRENT_SIZE = 512


class ExponentOutputLayer(nn.Module):
    def forward(self, input):
        return torch.exp(input)

class SquareOutputLayer(nn.Module):
    def forward(self, input):
        return torch.square(input)

class DirichletLayer(nn.Module):

    def forward(self, input):
        flat_input = torch.flatten(torch.exp(input), start_dim=2)
        sample = Dirichlet(flat_input).rsample()
        # multiplier = torch.diag(torch.tensor(CHANNEL_EXP_SUMS)).float()
        # scaled_sample = torch.matmul(multiplier, sample)
        # log_scaled_sample = torch.log(scaled_sample)
        # recovered = torch.reshape(log_scaled_sample, input.size())
        output = torch.reshape(sample, input.size())
        return output


# def populate_channel_exp_sum(img):
#     tt = img.reshape([3, 512 * 512])
#     out = []
#     for i in range(3):
#         ttt = tt[i]
#         channel_sum = sum(np.array([math.exp(xi) for xi in ttt]))
#         out.append(channel_sum)
#     global CHANNEL_EXP_SUMS
#     CHANNEL_EXP_SUMS = out

def get_softmax_gt(img_torch, img_np):
    t = torch.reshape(img_torch, [3, CURRENT_SIZE * CURRENT_SIZE])
    softmax_torch = torch.softmax(t, dim=1)
    softmax_torch = torch.reshape(softmax_torch, img_torch.size())
    tt = img_np.reshape([3, CURRENT_SIZE * CURRENT_SIZE])
    exp_sums = []
    for i in range(3):
        ttt = tt[i]
        channel_sum = sum(np.array([math.exp(xi) for xi in ttt]))
        exp_sums.append(channel_sum)
    return softmax_torch, exp_sums


def get_scaled_gt(img_torch, sigma):
    t = torch.reshape(img_torch, [3, CURRENT_SIZE * CURRENT_SIZE])
    print(t.is_cuda)
    sums = t.sum(1)
    print(sums.is_cuda)
    scaled = torch.matmul(torch.diag(1 / sums), t)
    scaled = torch.reshape(scaled, img_torch.size())
    print(scaled.is_cuda)
    return scaled, sigma / sums, sums


def recover_scale(img_np, sums):
    t = img_np.reshape([3, CURRENT_SIZE * CURRENT_SIZE])
    output = []
    for i in range(3):

        tt = t[i]
        cur_sum = sum(tt)
        prev_sum = sums[i]
        channel = np.array([xi * prev_sum / cur_sum for xi in tt])
        output.append(channel)
    output = np.array(output).reshape([3, CURRENT_SIZE, CURRENT_SIZE])
    return output

def recover_scale_torch(img_torch, sums):
    t = torch.reshape(img_torch, [3, CURRENT_SIZE * CURRENT_SIZE])
    cur_sums = t.sum(-1).reshape([3,1])
    out = t / cur_sums * (sums.reshape([3,1]))
    return torch.reshape(out, img_torch.size())


def recover_softmax(img_np, exp_sums, scalar):
    t = img_np.reshape([3, CURRENT_SIZE * CURRENT_SIZE])
    output = []
    for i in range(3):
        tt = t[i]
        exp_sum = exp_sums[i]
        channel = np.array([math.log(xi * exp_sum / scalar) for xi in tt])
        output.append(channel)
    output = np.array(output).reshape([3, CURRENT_SIZE, CURRENT_SIZE])
    return output

def sample(out, exp_sums):
    out_reshaped = torch.reshape(out, [3, CURRENT_SIZE * CURRENT_SIZE])
    dir = Dirichlet(out_reshaped)
    sampled = dir.sample()
    sampled_reshaped = torch.reshape(out, [3, CURRENT_SIZE, CURRENT_SIZE])
    return sampled_reshaped



def dirichlet_kl_divergence(p, q, scalar):
    q_reshaped = torch.reshape(q, [3, CURRENT_SIZE * CURRENT_SIZE]) * scalar
    p_reshaped = torch.reshape(p, [3, CURRENT_SIZE * CURRENT_SIZE])
    temp = kl_divergence(Dirichlet(p_reshaped), Dirichlet(q_reshaped)).sum()
    return temp.sum()

def pseudo_inverse_map(img_np, eps):
    tt = img_np.reshape([3, CURRENT_SIZE * CURRENT_SIZE])
    K = CURRENT_SIZE * CURRENT_SIZE
    out = []
    for i in range(3):
        cur_layer = tt[i]
        neg_exp_sum = sum(np.array([math.exp(-xi) for xi in cur_layer]))
        def transform(x):
            return (1 / eps) * (1 - 2 / K + math.exp(x) * neg_exp_sum / (K * K))
        vtransform = np.vectorize(transform)
        channel = vtransform(cur_layer)
        out.append(channel)
    return np.array(out).reshape([3, CURRENT_SIZE, CURRENT_SIZE])

def recover_bridge(img_np):
    t = img_np.reshape([3, CURRENT_SIZE * CURRENT_SIZE])
    output = []
    K = CURRENT_SIZE * CURRENT_SIZE
    for i in range(3):
        tt = t[i]
        log_sum = np.sum(np.array([math.log(xi) for xi in tt]))
        channel = np.array([math.log(xi) - (log_sum/K) for xi in tt])
        output.append(channel)
    output = np.array(output).reshape([3, CURRENT_SIZE, CURRENT_SIZE])
    return output

# alpha is the current param for dirichlet. Mean is the normal distribution's mean with sum(mean) = 1.
# def dirichlet_normal_kl_divergence_min_function(alpha, mean, std):
#     alpha = torch.reshape(alpha, [3, CURRENT_SIZE * CURRENT_SIZE])
#     mean = torch.reshape(mean, [3, CURRENT_SIZE * CURRENT_SIZE])
#     a0 = alpha.sum(-1)
#     k = CURRENT_SIZE * CURRENT_SIZE
#     entropy = (torch.lgamma(alpha).sum(-1) - torch.lgamma(a0) - (k - a0) * torch.digamma(a0) - ((alpha - 1.0) * torch.digamma(alpha)).sum(-1))
#     data_part = (0.5/(std*std)) * ((alpha * (alpha+1.0)).sum(-1) / (a0 * (a0 + 1.0)) - 2 * (mean * alpha).sum(-1) / a0)
#     return (-entropy + data_part).sum(-1)


class DirichletLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DirichletLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        alpha = inputs
        mean, std = targets
        alpha = torch.reshape(alpha, [3, CURRENT_SIZE * CURRENT_SIZE])
        mean = torch.reshape(mean, [3, CURRENT_SIZE * CURRENT_SIZE])
        a0 = alpha.sum(-1)
        k = CURRENT_SIZE * CURRENT_SIZE
        entropy = (torch.lgamma(alpha).sum(-1) - torch.lgamma(a0) - (k - a0) * torch.digamma(a0) - (
                    (alpha - 1.0) * torch.digamma(alpha)).sum(-1))
        data_part = (0.5 / (std * std)) * (
                    (alpha * (alpha + 1.0)).sum(-1) / (a0 * (a0 + 1.0)) - 2 * (mean * alpha).sum(-1) / a0)
        return (-entropy + data_part).sum(-1)
