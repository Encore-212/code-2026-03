import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # Work with broadcast, assuming batch as the first dimension
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    binary_mask = torch.floor(random_tensor)  # Create binary mask
    output = x.div(keep_prob) * binary_mask  # Scale the output
    return output


class DropPath(nn.Module):
    """DropPath Layer (Stochastic Depth)."""

    def __init__(self, drop_prob: float = 0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    #  assert mode in ["sum", "mul"]
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, mode="mul"):
        super().__init__()
        self.mode = mode
        self.norm = nn.LayerNorm(dim)

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        self.linear1 = nn.Linear(dim, 3 * dim)
        self.linear2 = nn.Linear(dim, 3 * dim)

        self.act = nn.GELU()
        self.g = nn.Linear(3 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else 1.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.norm(x)
        x = self.dwconv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        x = self.act(x1) * x2
        x = self.g(x)
        x = self.act(x)
        x = input + self.drop_path(self.gamma * x)
        return x




class SParallelCNN(nn.Module):
    def __init__(self, config):
        super(SParallelCNN, self).__init__()
        self.conv1_stream1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1_stream1 = nn.BatchNorm2d(16)
        self.conv2_stream1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2_stream1 = nn.BatchNorm2d(32)
        self.conv3_stream1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3_stream1 = nn.BatchNorm2d(64)
        self.pool_stream1 = nn.MaxPool2d(2, 2)
        self.dropout_stream1 = nn.Dropout(0.3)

        # Stream 2
        self.conv1_stream2 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1_stream2 = nn.BatchNorm2d(16)
        self.conv2_stream2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2_stream2 = nn.BatchNorm2d(32)
        self.conv3_stream2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3_stream2 = nn.BatchNorm2d(64)
        self.pool_stream2 = nn.MaxPool2d(2, 2)
        self.dropout_stream2 = nn.Dropout(0.3)

        self.norm = nn.LayerNorm(64)  # final norm layer
        self.block1 = Block(dim=64, drop_path=0.1, layer_scale_init_value=1e-6, mode="mul")
        self.block2 = Block(dim=64, drop_path=0.1, layer_scale_init_value=1e-6, mode="mul")

    def forward(self, x, return_stream=False):
        # Stream 1
        x1 = F.relu(self.conv1_stream1(x))
        x1 = self.bn1_stream1(x1)
        x1 = self.pool_stream1(x1)
        x1 = self.dropout_stream1(x1)

        x1 = F.relu(self.conv2_stream1(x1))
        x1 = self.bn2_stream1(x1)
        x1 = self.pool_stream1(x1)
        x1 = self.dropout_stream1(x1)

        x1 = F.relu(self.conv3_stream1(x1))
        x1 = self.bn3_stream1(x1)
        x1 = self.pool_stream1(x1)
        x1 = self.dropout_stream1(x1)
        # 输出维度：32,64,5,111
        # Stream 2
        x2 = F.relu(self.conv1_stream2(x))
        x2 = self.bn1_stream2(x2)
        x2 = self.pool_stream2(x2)
        x2 = self.dropout_stream2(x2)

        x2 = F.relu(self.conv2_stream2(x2))
        x2 = self.bn2_stream2(x2)
        x2 = self.pool_stream2(x2)
        x2 = self.dropout_stream2(x2)

        x2 = F.relu(self.conv3_stream2(x2))
        x2 = self.bn3_stream2(x2)
        x2 = self.pool_stream2(x2)
        x2 = self.dropout_stream2(x2)
        # 加入block模块 b,c,h, w ---> b, h, w, c
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        output1_block = self.block1(x1)
        # 只保留channel
        output1_x = self.norm(output1_block.mean([1, 2]))
        output2_block = self.block2(x2)
        output2_x = self.norm(output2_block.mean([1, 2]))
        # 形状 （32,128）
        x_combined = torch.cat((output1_x, output2_x), dim=1)
        if return_stream:
            '''
            方便双流计算余弦相似度
            '''
            return x_combined, output1_x, output2_x
        return x_combined


