import torch
import torch.nn as nn
import math
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class RAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_len):
        super(RAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_len = max_len
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim 必须能被 num_heads 整除"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # 相对位置编码参数，长度为 2*max_len-1
        self.relative_position = nn.Parameter(torch.empty(2 * max_len - 1))
        nn.init.normal_(self.relative_position, mean=0.0, std=0.02)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, mask=None, key_padding_mask=None):
        L, B, C = x.shape
        x = x.transpose(0, 1).contiguous()
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1))

        index_offset = torch.arange(L, device=x.device).unsqueeze(0) - torch.arange(L, device=x.device).unsqueeze(1)
        index_offset += self.max_len - 1
        R = self.relative_position[index_offset]
        R = R.unsqueeze(0).unsqueeze(0)
        R = R.expand(B, self.num_heads, L, L)

        #  (QK^T + R) / sqrt(d_k)
        attn_scores = attn_scores + R
        attn_scores = attn_scores * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).reshape(B, L, C)
        attn_output = attn_output.transpose(0, 1).contiguous()

        output = self.out_proj(attn_output)
        return output

class RTransformerEncoderLayer(nn.Module):
    def __init__(self, dim, heads=4, dim_feedforward=2048, dropout=0.1):
        super(RTransformerEncoderLayer, self).__init__()
        self.attention = RAttention(embed_dim=dim, num_heads=heads, max_len=225)
        # FFN
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        x = self.norm1(x + self.dropout(self.attention(x, mask=src_mask, key_padding_mask=src_key_padding_mask)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x




class RAudioTrans(nn.Module):
    def __init__(self, config):
        super(RAudioTrans, self).__init__()
        self.d_model = config['input_dim']
        self.nhead = config['heads']
        self.dim_feedforward = config['feedforward_dim']
        self.dropout = config['dropout']
        self.num_layers = config['transformer_layers']
        self.dropout_layer = nn.Dropout(self.dropout)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 4))
        self.pos_encoder = PositionalEncoding(self.d_model).to(device)
        self.createlayer = RTransformerEncoderLayer(self.d_model, self.nhead, self.dim_feedforward, self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.createlayer, self.num_layers)

    def forward(self, src, mask=None):
        src = src.squeeze()
        x_pool = self.maxpool(src)
        x_permute = x_pool.squeeze().permute(2, 0, 1)
        x_pos = self.pos_encoder(x_permute)
        transformer_out = self.transformer_encoder(x_pos, mask=mask)
        transformer_out = transformer_out.mean(dim=0)
        return transformer_out
