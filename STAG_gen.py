#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import argparse
from scipy.optimize import linprog

np.seterr(divide='ignore', invalid='ignore')


def wasserstein_distance(p, q, D):
    # 存储约束矩阵
    A_eq = []
    for i in range(len(p)):
        # create an array thay with the same shape and data type as D
        A = np.zeros_like(D)
        A[i, :] = 1
        # reshape中-1是占位符 整个为将其reshape为一维数组
        A_eq.append(A.reshape(-1))
        # len of A_eq is 35
        # print(f'A_eq:{len(A_eq)}')
    for i in range(len(q)):
        A = np.zeros_like(D)
        A[:, i] = 1
        A_eq.append(A.reshape(-1))
        # len of A_eq is 70
        # print(f'A_eq:{len(A_eq)}')
    A_eq = np.array(A_eq)
    # A_eq:(70, 1225)
    # print(f'A_eq:{A_eq.shape}')
    b_eq = np.concatenate([p, q])
    # A_eq:(70, 1225)
    # b_eq:(70,) 一维数组
    # print(f'b_eq:{b_eq.shape}')
    D = np.array(D)
    D = D.reshape(-1)
    # 线性规划问题
    result = linprog(D, A_eq=A_eq[:-1], b_eq=b_eq[:-1])
    myresult = result.fun
    # print(f'myresult:{myresult}')
    return myresult


def spatial_temporal_aware_distance(x, y):
    print('spatial_temporal_aware_distance is ok')
    x, y = np.array(x), np.array(y)
    # 归一化处理 计算L2范数 X=（x1,x2,...x35）
    x_norm = (x ** 2).sum(axis=1, keepdims=True) ** 0.5
    y_norm = (y ** 2).sum(axis=1, keepdims=True) ** 0.5
    # print(f'x_norm.shape:{x_norm.shape}')
    # x_norm.shape: (35, 1) 每一天的
    # 得到归一化的概率分布向量 p 和 q
    p = x_norm[:, 0] / x_norm.sum()
    # print(f'p.shape:{p.shape}')
    q = y_norm[:, 0] / y_norm.sum()
    # .T是转置 D表示x与y样本距离矩阵
    D = 1 - np.dot(x / x_norm, (y / y_norm).T)
    # print(f'D.shape:{D.shape}') 35 x 35
    # Check if D contains invalid values
    # if np.any(np.isinf(D)) or np.any(np.isnan(D)) or D is None:
    #     # Replace invalid values with a valid value, e.g., 0
    #     D[np.isinf(D)] = 0
    #     D[np.isnan(D)] = 0
    if np.any(np.isinf(D)) or np.any(np.isnan(D)):
        # Replace invalid values with a valid value, e.g., 0
        D[np.isinf(D)] = 0
        D[np.isnan(D)] = 0
    return wasserstein_distance(p, q, D)


def spatial_temporal_similarity(x, y, normal, transpose):
    if normal:
        x = normalize(x)
        y = normalize(y)
    if transpose:
        x = np.transpose(x)
        y = np.transpose(y)
    print('spatial_temporal_similarity is ok')
    return 1 - spatial_temporal_aware_distance(x, y)


def gen_data(data, ntr, N):
    """
    if flag:
        data=pd.read_csv(fname)
    else:
        data=pd.read_csv(fname,header=None)
    """
    data = np.reshape(data, [-1, 288, N])
    return data[0:ntr]


def normalize(a):
    mu = np.mean(a, axis=1, keepdims=True)
    std = np.std(a, axis=1, keepdims=True)
    return (a - mu) / std


# 命令行参数解析器
parser = argparse.ArgumentParser()
# 数据集路径的参数
parser.add_argument("--dataset", type=str, default="PEMS08", help="Dataset path.")
# 时间序列周期的参数
parser.add_argument("--period", type=int, default=288, help="Time series periods.")
# 空间图的稀疏度参数
parser.add_argument("--sparsity", type=float, default=0.01, help="sparsity of spatial graph")

args = parser.parse_args()
print(f'args.dataset:{args.dataset}')
df = np.load(args.dataset + '/' + args.dataset + ".npz")['data']
# print(f'df_shape:{df.shape}')
# 总的时间点数 节点个数 忽略第三维
num_samples, ndim, _ = df.shape
# print(f'num_samples:{num_samples}')
# print(f'ndim:{ndim}')
num_train = int(num_samples * 0.6)
# 计算一个以时间步长period为周期的训练集样本数量 向下取整后再乘保证训练集样本数量是时间步长的整数倍
num_sta = int(num_train / args.period) * args.period
data = df[:num_sta, :, :1].reshape([-1, args.period, ndim])
# PEMS04中为35 * 288 * 307 [[[...307],...,[...307]288个],...,[[]]35个]
# 最内层为第几天何时的一个传感器的数据
print(f'data_shape:{data.shape}')

d = np.zeros([ndim, ndim])
t0 = time.time()
# print(f't0:{t0}')
for i in range(ndim):
    # print('in range')
    t1 = time.time()
    for j in range(i + 1, ndim):
        # data_shape:(35, 288, 307)
        # data[:, :, i].shape:(35, 288)
        # print(f'data[:, :, i].shape:{data[:, :, i].shape}')
        d[i, j] = spatial_temporal_similarity(data[:, :, i], data[:, :, j], normal=False, transpose=False)
        # d[i, j] = spatial_temporal_similarity(data[:, :, i], data[:, :, j], normal=True, transpose=True)
        print('start:\t', j, end='', flush=True)
    t2 = time.time()
    print('Line', i, 'finished in', t2 - t1, 'seconds.')

# 初步理解是无向图所以加
# 时空感知距离图
sta = d + d.T

np.save(args.dataset + '/' + "stag_001_" + args.dataset + ".npy", sta)
print("The calculation of time series is done!")
t3 = time.time()
print('total finished in', t3 - t0, 'seconds.')
adj = np.load(args.dataset + '/' + "stag_001_" + args.dataset + ".npy")
# 单位矩阵
id_mat = np.identity(ndim)
adjl = adj + id_mat
# 沿列方向进行归一化处理，即除以列的平均值
adjlnormd = adjl / adjl.mean(axis=0)

adj = 1 - adjl + id_mat
A_adj = np.zeros([ndim, ndim])
R_adj = np.zeros([ndim, ndim])
# A_adj = adj
adj_percent = args.sparsity

top = int(ndim * adj_percent)

# 对当前行进行排序，选取前 top 个最小的元素的索引
for i in range(adj.shape[0]):
    a = adj[i, :].argsort()[0:top]
    for j in range(top):
        # 二值化时空相关图后A_adj是时空感知图 aware graph 最相关节点邻接矩阵
        A_adj[i, a[j]] = 1
        # R_adj是时空相关图 relevance graph 也就是权重矩阵
        R_adj[i, a[j]] = adjlnormd[i, a[j]]

for i in range(ndim):
    for j in range(ndim):
        if i == j:
            R_adj[i][j] = adjlnormd[i, j]

print("Total route number: ", ndim)
# 稀疏度
print("Sparsity of adj: ", len(A_adj.nonzero()[0]) / (ndim * ndim))

pd.DataFrame(A_adj).to_csv(args.dataset + '/' + "stag_001_" + args.dataset + ".csv", index=False, header=None)
pd.DataFrame(R_adj).to_csv(args.dataset + '/' + "strg_001_" + args.dataset + ".csv", index=False, header=None)

print("The weighted matrix of temporal graph is generated!")
