import torch.nn
from torch import nn
import math
from torch.distributions.dirichlet import *
import numpy as np

CHANNEL_EXP_SUMS = [1,1,1]

class DirichletLayer(nn.Module):

    def forward(self, input):
        flat_input = torch.flatten(input, start_dim=2)
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

def get_softmax_gt(img_np, img_torch):
    t = torch.reshape(img_torch, [3, 512 * 512])
    softmax_torch = torch.softmax(t, dim=1)
    softmax_torch = torch.reshape(softmax_torch, img_torch.size())
    tt = img_np.reshape([3, 512 * 512])
    exp_sums = []
    for i in range(3):
        ttt = tt[i]
        channel_sum = sum(np.array([math.exp(xi) for xi in ttt]))
        exp_sums.append(channel_sum)
    return softmax_torch, exp_sums

def recover(img_np, exp_sums):

    t = img_np.reshape([3, 512 * 512])
    output = []
    for i in range(3):
        tt = t[i]
        exp_sum = exp_sums[i]
        channel = np.array([math.log(xi * exp_sum) for xi in tt])
        output.append(channel)
    output = np.array(output).reshape([3, 512, 512])
    return output

