from argparse import ArgumentParser
import numpy as np
import torch
from data import get_dataset, DATASET_CONFIGS
from train import train
from model import MLP
import utils

import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
import utils
from torch.nn import functional as F
#
# x = torch.randn(2,3)
# x = x **2
# print(x)
# # x = x.unbind()
# print(x.mean(0))
#
# x = torch.tensor([[1,2,5,3],[5,6,5,6]],dtype=torch.float32)
# y = torch.tensor([2,3],dtype=torch.int64)
# # z =torch.cat((x,y),dim=0)
# # print(F.softmax(z,dim=0))
#
# z = F.log_softmax(x, dim=1)#[range(0,2), y]
# # print(z)
# # # print(z[y])
# # print(z[range(1)])
# # print(z[range(2),y])
# # print(z[0][2])
#
# z = torch.tensor([[1,2,5,3],[5,6,5,6]],dtype=torch.float32)
# out = torch.cat((x,x,z),dim =0).unbind()
# # print(out[0].type())




# a = [1,2]
# out = a**2
# print(out)
# b = [3,4]
# c = zip(a,b)
# # tmp = zip(*c)
# print(dict(tmp))
# k =1
# for x in out:
#     torch.stack((x),dim=0)
# print(x.shape)


permutations = [
    np.random.permutation(DATASET_CONFIGS['mnist']['size'] ** 2) for
    _ in range(2)
]

train_datasets = [
    get_dataset('mnist', permutation=p) for p in permutations
]
test_datasets = [
    get_dataset('mnist', train=False, permutation=p) for p in permutations
]
mlp = MLP(
        DATASET_CONFIGS['mnist']['size']**2,
        DATASET_CONFIGS['mnist']['classes'],
        hidden_size=20,
        hidden_layer_num=2,
        hidden_dropout_prob=.5,
        input_dropout_prob=.2,
        lamda=40,
    )


k = mlp.estimate_fisher( train_datasets[0], 1024)

# out = mlp.mytest(1)
# print(out)

# print(k)