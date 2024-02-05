# -*- coding:utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.utils import scaled_Laplacian, cheb_polynomial, cheb_polynomial_org

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    print('cuda')
    DEVICE = torch.device('cuda')
else:
    print('cpu')
    DEVICE = torch.device('cpu')

print("model_CUDA:", USE_CUDA, DEVICE)


class SScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(SScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, attn_mask):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
        return scores


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, num_of_d):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.num_of_d = num_of_d

    def forward(self, Q, K, V, attn_mask, res_att):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.d_k) + res_att  # scores : [batch_size, n_heads, len_q, len_k]
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
        # (32, 1, 3, 12, 12)
        attn = F.softmax(scores, dim=3)
        # (32, 1, 3, 12, 32)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, scores


class SMultiHeadAttention(nn.Module):
    def __init__(self, DEVICE, d_model, d_k, d_v, n_heads):
        super(SMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.DEVICE = DEVICE
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)

    def forward(self, input_Q, input_K, attn_mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                     2)  # Q: [batch_size, n_heads,
        # len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                     2)  # K: [batch_size, n_heads,
        # len_k, d_k]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                      1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        attn = SScaledDotProductAttention(self.d_k)(Q, K, attn_mask)
        return attn


class MultiHeadAttention(nn.Module):
    def __init__(self, DEVICE, d_model, d_k, d_v, n_heads, num_of_d):
        super(MultiHeadAttention, self).__init__()
        # d_model:307
        self.d_model = d_model
        # d_k:32
        self.d_k = d_k
        # d_v: 32
        self.d_v = d_v
        # n_heads: 3
        self.n_heads = n_heads
        # num_of_d： 32
        self.num_of_d = num_of_d
        self.DEVICE = DEVICE
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask, res_att):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_k).transpose(2,
                                                                                                    3)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_k).transpose(2,
                                                                                                    3)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_v).transpose(2,
                                                                                                    3)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                      1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, res_attn = ScaledDotProductAttention(self.d_k, self.num_of_d)(Q, K, V, attn_mask, res_att)

        context = context.transpose(2, 3).reshape(batch_size, self.num_of_d, -1,
                                                  self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        # （32， 1， 12， 307）
        output = self.fc(context)  # [batch_size, len_q, d_model]

        return nn.LayerNorm(self.d_model).to(self.DEVICE)(output + residual), res_attn


class cheb_conv_withSAt(nn.Module):
    """
    K-order chebyshev graph convolution
    """

    def __init__(self, K, cheb_polynomials, in_channels, out_channels, num_of_vertices):
        """
        :param K: int 切比雪夫多项式项数
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        """
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU()
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])
        self.mask = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x, spatial_attention, adj_pa):
        """
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        :adj_pa:
        :spatial_attention:

        """

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):
                T_k = self.cheb_polynomials[k]  # (N,N)
                mask = self.mask[k]

                myspatial_attention = spatial_attention[:, k, :, :] + adj_pa.mul(mask)
                myspatial_attention = F.softmax(myspatial_attention, dim=1)

                # 将TA和SA并行 门控融合

                T_k_with_at = T_k.mul(myspatial_attention)  # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = T_k_with_at.permute(0, 2, 1).matmul(
                    graph_signal)  # (N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘

                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return self.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)


class cheb_conv(nn.Module):
    """
    K-order chebyshev graph convolution
    """

    def __init__(self, K, cheb_polynomials_org, in_channels, out_channels):
        """
        :param K: int        :param in_channles: , num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        """
        super(cheb_conv, self).__init__()
        self.K = K
        self.cheb_polynomials_org = cheb_polynomials_org
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(inplace=True)
        self.DEVICE = cheb_polynomials_org[0].device
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x):
        """
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        """

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):
                T_k = self.cheb_polynomials_org[k]  # (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)
                # rhs = graph_signal.permute(0, 2, 1).matmul(T_k)

                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return self.relu(torch.cat(outputs, dim=-1))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.05, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, X):
        X = X + self.pe[:X.size(0)]
        X = self.dropout(X)
        return X


class Embedding(nn.Module):
    def __init__(self, nb_seq, d_Em, num_of_features, Etype):
        # d_Em: 嵌入向量的维度
        # Etype: 嵌入类型
        super(Embedding, self).__init__()
        # 表示输入序列的长度（序列的时间步数）
        self.nb_seq = nb_seq
        self.Etype = Etype
        # 表示输入数据的特征数量（在时间步上的通道数）1
        self.num_of_features = num_of_features

        """pos_embed: Embedding(12, 307) 
        其中nb_seq是输入序列的长度，d_Em是嵌入向量的维度
        该层用于将输入的离散序列索引嵌入到连续的实数向量空间。
        """
        self.pos_embed = nn.Embedding(nb_seq, d_Em)

        self.norm = nn.LayerNorm(d_Em)

    def forward(self, x, batch_size):
        if self.Etype == 'T':
            pos = torch.arange(self.nb_seq, dtype=torch.long).to(DEVICE)
            pos = pos.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_of_features,
                                                       self.nb_seq)  # [seq_len] -> [batch_size, seq_len]
            embedding = x.permute(0, 2, 3, 1) + self.pos_embed(pos)
        else:
            pos = torch.arange(self.nb_seq, dtype=torch.long).to(DEVICE)
            pos = pos.unsqueeze(0).expand(batch_size, self.nb_seq)
            embedding = x + self.pos_embed(pos)
        Emx = self.norm(embedding)
        return Emx


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=None, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, X):
        X = X + self.pe[:X.size(0)]
        X = self.dropout(X)
        return X


class GTU(nn.Module):
    def __init__(self, in_channels, time_strides, kernel_size):
        super(GTU, self).__init__()
        self.in_channels = in_channels
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU(inplace=True)
        self.con2out = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=(1, kernel_size), stride=(1, time_strides))

    def forward(self, x):
        # x:(32, 32, 307, 12)
        # (32, 64, 307, 10)
        x_causal_conv = self.con2out(x)
        # (32, 32, 307, 10)
        x_p = x_causal_conv[:, : self.in_channels, :, :]
        x_q = x_causal_conv[:, -self.in_channels:, :, :]
        x_gtu = torch.mul(self.tanh(x_p), self.sigmoid(x_q))
        # x_gtu = torch.mul(self.tanh(x_p), self.relu(x_q))
        # x_gtu = self.tanh(x_p)
        return x_gtu


class SFusion(nn.Module):
    def __init__(self, in_channels):
        super(SFusion, self).__init__()
        # nn.Parameter 是一种特殊的张量类型，用于标记模型参数，使其在模型优化过程中可以自动进行更新（学习）
        # 确保 alpha 在合理的范围内
        # x1是动态输出
        self.in_channels = in_channels
        self.alpha = nn.Parameter(torch.rand(1))
        # self.alpha = torch.clamp(self.alpha, 0.0, 1.0)

        """
        self.weight_pa = nn.Parameter(torch.rand(1))
        self.weight_org = nn.Parameter(torch.rand(1))
        self.relu = nn.ReLU()
        """
        self.relu = nn.ReLU()
        # self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x1, x2):
        """
        # x1, x2: [Batch_size, in_channels, sequence_length]
        fused_output = self.weight_pa * x1 + self.weight_org * x2
        fused_output = self.relu(fused_output)
        return fused_output
        """
        # 根据 alpha 对 x1 和 x2 进行加权求和
        # fused_output = 0.8 * x1 + 0.2 * x2
        fused_output = self.alpha * x1 + (1 - self.alpha) * x2
        fused_output = self.relu(fused_output)

        return fused_output


class TFusion(nn.Module):
    def __init__(self, in_channels):
        super(TFusion, self).__init__()
        # nn.Parameter 是一种特殊的张量类型，用于标记模型参数，使其在模型优化过程中可以自动进行更新（学习）
        # 确保 alpha 在合理的范围内
        self.in_channels = in_channels
        self.alpha = nn.Parameter(torch.rand(1))
        # self.alpha = torch.clamp(self.alpha, 0.0, 1.0)

    def forward(self, x1, x2):
        # 根据 alpha 对 x1 和 x2 进行加权求和
        fused_output = self.alpha * x1 + (1 - self.alpha) * x2

        return fused_output


class STFusion(nn.Module):
    def __init__(self, in_channels):
        super(STFusion, self).__init__()
        # nn.Parameter 是一种特殊的张量类型，用于标记模型参数，使其在模型优化过程中可以自动进行更新（学习）
        # 确保 alpha 在合理的范围内
        # x1是动态输出
        self.in_channels = in_channels
        self.alpha = nn.Parameter(torch.rand(1))
        # self.alpha = torch.clamp(self.alpha, 0.0, 1.0)

        """
        self.weight_pa = nn.Parameter(torch.rand(1))
        self.weight_org = nn.Parameter(torch.rand(1))
        self.relu = nn.ReLU()
        """
        self.relu = nn.ReLU()
        # self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x1, x2):
        """
        # x1, x2: [Batch_size, in_channels, sequence_length]
        fused_output = self.weight_pa * x1 + self.weight_org * x2
        fused_output = self.relu(fused_output)
        return fused_output
        """
        # 根据 alpha 对 x1 和 x2 进行加权求和
        # fused_output = 0.8 * x1 + 0.2 * x2
        fused_output = self.alpha * x1 + (1 - self.alpha) * x2
        fused_output = self.relu(fused_output)

        return fused_output


class ChebFusion(nn.Module):
    def __init__(self, in_channels):
        super(ChebFusion, self).__init__()
        # nn.Parameter 是一种特殊的张量类型，用于标记模型参数，使其在模型优化过程中可以自动进行更新（学习）
        # 确保 alpha 在合理的范围内
        self.in_channels = in_channels
        self.weight_1 = nn.Parameter(torch.rand(1))
        self.weight_2 = nn.Parameter(torch.rand(1))
        self.weight_3 = nn.Parameter(torch.rand(1))
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        # x1, x2: [Batch_size, in_channels, sequence_length]
        fused_output = self.weight_1 * x1 + self.weight_2 * x2 + self.weight_3 * x3
        fused_output = self.relu(fused_output)
        return fused_output


class Temporal_Attention_layer(nn.Module):
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))

    def forward(self, x):
        """
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        """
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)

        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)

        E_normalized = F.softmax(E, dim=1)

        return E_normalized


# 8.4加 使用gru
class RNNLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        # [batch_size, seq_len, num_nodes, hidden_dim] = X.shape
        [batch_size, hidden_dim, num_nodes, seq_len] = X.shape
        X = X.transpose(1, 2).reshape(batch_size * num_nodes, seq_len, hidden_dim)
        hx = torch.zeros_like(X[:, 0, :])
        output = []
        for _ in range(X.shape[1]):
            hx = self.gru_cell(X[:, _, :], hx)
            output.append(hx)
        output = torch.stack(output, dim=0)
        # output = self.dropout(output)
        return output


class DSTAGNN_block(nn.Module):

    def __init__(self, DEVICE, num_of_d, in_channels, K, nb_chev_filter, nb_time_filter, time_strides,
                 cheb_polynomials_org, cheb_polynomials, adj_pa, adj_TMD, adj_org, num_of_vertices, num_of_timesteps,
                 d_model, d_k, d_v, n_heads):
        super(DSTAGNN_block, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # 原地计算张量
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(inplace=True)

        # 将numpy二维数组转换为浮点数型张量
        self.adj_pa = torch.FloatTensor(adj_pa).to(DEVICE)
        # （12，512）kernel_size(1, 1) stride(1, 1)
        self.pre_conv = nn.Conv2d(num_of_timesteps, d_model, kernel_size=(1, num_of_d))
        self.EmbedT = Embedding(num_of_timesteps, num_of_vertices, num_of_d, 'T')
        self.EmbedT_for_time = Embedding(num_of_timesteps, num_of_vertices, 32, 'T')
        # d_model:512 (307,512)
        self.EmbedS = Embedding(num_of_vertices, d_model, num_of_d, 'S')
        # in: 307 out: 96 fc_in:96 fc_out:307
        # 没用dmodel?
        self.TAt = MultiHeadAttention(DEVICE, num_of_vertices, d_k, d_v, n_heads, num_of_d)
        self.TAt_for_Sc = Temporal_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        # in :512 out:96
        self.SAt = SMultiHeadAttention(DEVICE, d_model, d_k, d_v, K)

        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter, num_of_vertices)
        # 使用静态图卷积
        self.cheb_conv1 = cheb_conv(1, cheb_polynomials_org, in_channels, nb_chev_filter)
        self.cheb_conv2 = cheb_conv(2, cheb_polynomials_org, in_channels, nb_chev_filter)
        self.cheb_conv3 = cheb_conv(3, cheb_polynomials_org, in_channels, nb_chev_filter)
        self.ChebFusion = ChebFusion(in_channels)
        self.SFusion = SFusion(in_channels)
        self.TFusion = TFusion(in_channels)
        self.STFusion = STFusion(in_channels)

        self.gtu3 = GTU(nb_time_filter, time_strides, 2)
        self.gtu5 = GTU(nb_time_filter, time_strides, 5)
        self.gtu7 = GTU(nb_time_filter, time_strides, 8)

        self.gru = RNNLayer(32)

        # stride: (1,2)
        self.pooling = torch.nn.MaxPool2d(kernel_size=(1, 2), stride=None, padding=0,
                                          return_indices=False, ceil_mode=False)
        # self.pooling = torch.nn.AvgPool2d(kernel_size=(1, 2), stride=None, padding=0,
        #                                 ceil_mode=False)
        # in_channels: 32 nb_time_filter: 32
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))

        self.dropout = nn.Dropout(p=0.05)
        self.fcmy = nn.Sequential(
            nn.Linear(3 * num_of_timesteps - 12, num_of_timesteps),
            nn.Dropout(0.05),
        )
        self.ln = nn.LayerNorm(nb_time_filter)
        self.linear_layer = nn.Linear(1, 32)

    # def forward(self, x, res_tatt, res_satt):
    def forward(self, x, res_tatt):
        """
        :param x: (Batch_size, N, F_in, T)
        :param res_tatt: (Batch_size, N, F_in, T)
        :param res_satt: (Batch_size, N, F_in, T)
        :return: (Batch_size, N, nb_time_filter, T)

        """
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape  # B,N,F,T

        # TAT
        if num_of_features == 1:
            # (32, 1, 12, 307)
            TEmx = self.EmbedT(x, batch_size)  # B,F,T,N
        else:
            TEmx = x.permute(0, 2, 3, 1)
        # 将 TEmx 作为查询、键、值，进行自注意力计算，得到 TATout(context) 和 re_At（dot后的scores）
        # TATout, TAToutput, re_At = self.TAt(TEmx, TEmx, TEmx, None, res_tatt)  # B,F,T,N; B,F,Ht,T,T
        TATout, re_At = self.TAt(TEmx, TEmx, TEmx, None, res_tatt)  # B,F,T,N; B,F,Ht,T,T

        # 7.31conv前加attention
        # temporal_At = torch.mean(re_At, dim=2).squeeze()
        # print(f'temporal_At.shape():{temporal_At.shape}')
        # x_TA = torch.matmul(x, temporal_At)
        temporal_At = self.TAt_for_Sc(x)
        # print(f'temporal_At.shape():{temporal_At.shape}')
        x_TAt_output = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At).reshape(batch_size,
                                                                                                      num_of_vertices,
                                                                                                      num_of_features,
                                                                                                      num_of_timesteps)


        # 将第三维从 1 映射到 32
        # (32, 307, 12, 32)
        # x_TAt_output = self.linear_layer(x_TAt_output.permute(0, 1, 3, 2)).permute(0, 3, 1, 2)

        # SAT 7.27加
        # if num_of_features == 1:
        #     SEmx_1 = self.EmbedT(x, batch_size)  # B,F,T,N
        # else:
        #     SEmx_1 = x.permute(0, 2, 3, 1)
        # 将 TEmx 作为查询、键、值，进行自注意力计算，得到 TATout(context) 和 re_At（dot后的scores）
        # SATout, re_SAt = self.TAt(SEmx_1, SEmx_1, SEmx_1, None, res_satt)  # B,F,T,N; B,F,Ht,T,T
        # Xout = self.SFusion(TATout, SATout)

        # SAt
        # 对 TATout 进行 1x1 卷积操作，得到 x_TAt
        # （32, 1， 307， 512）
        x_TAt = self.pre_conv(TATout.permute(0, 2, 3, 1))[:, :, :, -1].permute(0, 2, 1)  # B,N,d_model
        SEmx_TAt = self.EmbedS(x_TAt, batch_size)  # B,N,d_model
        SEmx_TAt = self.dropout(SEmx_TAt)  # B,N,d_model
        # 将 SEmx_TAt 作为查询、键，执行自注意力计算，得到 STAt
        # STAt: (32, 3, 307, 307)
        STAt = self.SAt(SEmx_TAt, SEmx_TAt, None)  # B,Hs,N,N

        # graph convolution in spatial dim
        # 使用邻接矩阵 adj_pa 执行图卷积操作
        # spatial_gcn_pa: (32, 307, 32, 12)
        spatial_gcn_pa = self.cheb_conv_SAt(x, STAt, self.adj_pa)  # B,N,F,T
        # spatial_gcn_pa = self.cheb_conv_SAt(Xout.permute(0, 3, 1, 2), STAt, self.adj_pa)  # B,N,F,T
        # spatial_gcn_pa = self.cheb_conv_SAt(TAToutput.permute(0, 3, 1, 2), STAt, self.adj_pa)  # B,N,F,T

        # spatial_gcn_org: (32, 307, 32, 12)
        spatial_gcn_org_1 = self.cheb_conv1(x_TAt_output)
        # spatial_gcn_org_2 = self.cheb_conv2(x_TAt_output)
        # spatial_gcn_org_3 = self.cheb_conv3(x_TAt_output)

        # spatial_gcn_org = self.ChebFusion(spatial_gcn_org_1, spatial_gcn_org_2, spatial_gcn_org_3)

        X_pa = spatial_gcn_pa.permute(0, 2, 1, 3)  # B,F,N,T
        X_org = spatial_gcn_org_1.permute(0, 2, 1, 3)  # B,F,N,T
        # 并行卷积 将结果进行融合
        X = self.SFusion(X_pa, X_org)

        # X_for_time = self.EmbedT_for_time(x, batch_size).permute(0, 1, 3, 2)

        # gru_output = self.gru(x_TAt_output)
        gru_output = self.gru(X)
        # gru_output = self.gru(time_conv_output)
        # gru_output = self.gru(time_conv)
        gru_output = gru_output.reshape(num_of_timesteps, batch_size, num_of_vertices, 32).permute(1, 3, 2, 0)

        # convolution along the time axis
        # 其中的每个是（32, 32, 307, 10/8/6)
        # x_gtu = [self.gtu3(X_for_time), self.gtu5(X_for_time), self.gtu7(X_for_time), gru_output]
        x_gtu = [self.gtu3(X), self.gtu5(X), self.gtu7(X)]
        # x_gtu = [self.gtu3(X), self.gtu5(X), self.gtu7(X)]
        time_conv = torch.cat(x_gtu, dim=-1)  # B,F,N,3T-12
        # (32, 32, 307, 12)
        time_conv = self.fcmy(time_conv)

        if num_of_features == 1:
            # (32, 32, 307, 12)
            time_conv_output = self.relu(time_conv)
        else:
            time_conv_output = self.relu(X + time_conv)  # B,F,N,T

        gru_output = self.TFusion(time_conv_output, gru_output)

        # output = self.STFusion(X, gru_output)

        # residual shortcut
        if num_of_features == 1:
            # (32, 307, 32, 12)
            x_residual = self.residual_conv(x.permute(0, 2, 1, 3))
        else:
            x_residual = x.permute(0, 2, 1, 3)
        # x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        x_residual = self.ln(F.relu(x_residual + gru_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        # x_residual = self.ln(F.relu(x_residual + output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)

        # return x_residual, re_At, re_SAt
        return x_residual, re_At


class DSTAGNN_submodule(nn.Module):

    def __init__(self, DEVICE, num_of_d, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides,
                 cheb_polynomials_org, cheb_polynomials, adj_pa, adj_TMD, adj_org, num_for_predict, len_input,
                 num_of_vertices, d_model, d_k, d_v,
                 n_heads):
        """
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param num_for_predict:
        """

        super(DSTAGNN_submodule, self).__init__()

        # 创建了一个 nn.ModuleList 类型的列表 其中包含了一个 DSTAGNN_block 的实例
        self.BlockList = nn.ModuleList([DSTAGNN_block(DEVICE, num_of_d, in_channels, K,
                                                      nb_chev_filter, nb_time_filter, time_strides,
                                                      cheb_polynomials_org, cheb_polynomials,
                                                      adj_pa, adj_TMD, adj_org, num_of_vertices, len_input, d_model,
                                                      d_k, d_v,
                                                      n_heads)])

        # 向容器中加入多个DSTAGNN_block实例 共nb_block个
        self.BlockList.extend([DSTAGNN_block(DEVICE, num_of_d * nb_time_filter, nb_chev_filter, K,
                                             nb_chev_filter, nb_time_filter, 1, cheb_polynomials_org, cheb_polynomials,
                                             adj_pa, adj_TMD, adj_org, num_of_vertices, len_input // time_strides,
                                             d_model, d_k,
                                             d_v, n_heads) for _ in range(nb_block - 1)])

        self.final_conv = nn.Conv2d(int((len_input / time_strides) * nb_block), 128, kernel_size=(1, nb_time_filter))
        self.final_fc = nn.Linear(128, num_for_predict)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x):
        """
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        """
        """
        for block in self.BlockList:
            x = block(x)

        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        """
        need_concat = []
        res_tatt = 0
        # res_satt = 0
        for block in self.BlockList:
            # x, res_tatt, res_satt = block(x, res_tatt, res_satt)
            x, res_tatt = block(x, res_tatt)
            need_concat.append(x)

        final_x = torch.cat(need_concat, dim=-1)
        output1 = self.final_conv(final_x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        output = self.final_fc(output1)

        return output


def make_model(DEVICE, num_of_d, nb_block, in_channels, K,
               nb_chev_filter, nb_time_filter, time_strides, adj_mx, adj_pa,
               adj_TMD, adj_org, num_for_predict, len_input, num_of_vertices, d_model, d_k, d_v, n_heads):
    """

    :param DEVICE:
    :param nb_block:
    :param in_channels:
    :param K:
    :param nb_chev_filter:
    :param nb_time_filter:
    :param time_strides:
    :param num_for_predict:
    :param len_input
    :return:
    """
    # 计算图的标准化拉普拉斯矩阵
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)]

    L_tilde_org = scaled_Laplacian(adj_org)
    cheb_polynomials_org = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in
                            cheb_polynomial_org(L_tilde_org, 1)]

    model = DSTAGNN_submodule(DEVICE, num_of_d, nb_block, in_channels,
                              K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials_org, cheb_polynomials,
                              adj_pa, adj_TMD, adj_org, num_for_predict, len_input, num_of_vertices, d_model, d_k, d_v,
                              n_heads)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model
