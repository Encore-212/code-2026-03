import torch
import torch.nn as nn
import torch.nn.functional as F

from cnn_star_multistream_model import StarParallelCNN
from model_transformer import RPEAudioTrans
from model_transformer import AbsAudioTrans
from cnn_star_differentfuse_model import StarAddCNN,GaussianCNN,StarConcatCNN,OuterProductCNN



'''
cnn分支不同
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AudioStarRpe(nn.Module):
    def __init__(self, config):
        super(AudioStarRpe, self).__init__()
        # Configuration parameters
        self.num_classes = config['num_classes']
        self.dropout_layer = nn.Dropout(config['dropout'])
        # return_stream开关是否计算双流
        self.cnnout = StarParallelCNN(config)
        self.transformerout = RPEAudioTrans(config)

        # Linear Classifier
        self.fc1 = nn.Linear(in_features=128 + config['input_dim'], out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=self.num_classes)

    def forward(self, x):
        x = x.to(device)
        # 拼接两个路径的输出
        cnn_out = self.cnnout(x)
        transformer_out = self.transformerout(x)
        # Fusion and classification
        combined_features = torch.cat((cnn_out, transformer_out), dim=1)
        # 通过两层全连接网络
        out = self.fc1(combined_features)
        out = self.dropout_layer(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out

class AudioStarAbs(nn.Module):
    def __init__(self, config):
        super(AudioStarAbs, self).__init__()
        # Configuration parameters
        self.num_classes = config['num_classes']
        self.dropout_layer = nn.Dropout(config['dropout'])
        self.cnnout = StarParallelCNN(config)
        self.transformerout = AbsAudioTrans(config)

        # Linear Classifier
        self.fc1 = nn.Linear(in_features=128 + config['input_dim'], out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=self.num_classes)

    def forward(self, x):
        x = x.to(device)
        # 拼接两个路径的输出
        cnn_out = self.cnnout(x)
        transformer_out = self.transformerout(x)
        # Fusion and classification
        combined_features = torch.cat((cnn_out, transformer_out), dim=1)
        # 通过两层全连接网络
        out = self.fc1(combined_features)
        out = self.dropout_layer(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out

class StarAddRpe(nn.Module):
    def __init__(self, config):
        super(StarAddRpe, self).__init__()
        # Configuration parameters
        self.num_classes = config['num_classes']
        self.dropout_layer = nn.Dropout(config['dropout'])
        self.cnnout = StarAddCNN(config)
        self.transformerout = RPEAudioTrans(config)

        # Linear Classifier
        self.fc1 = nn.Linear(in_features=128 + config['input_dim'], out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=self.num_classes)

    def forward(self, x):
        x = x.to(device)
        # 为cmdc加的
        # x = x.unsqueeze(1)
        # x = x.permute(0, 1, 3, 2)
        # 拼接两个路径的输出
        cnn_out = self.cnnout(x)
        transformer_out = self.transformerout(x)
        # Fusion and classification
        combined_features = torch.cat((cnn_out, transformer_out), dim=1)
        # 通过两层全连接网络
        out = self.fc1(combined_features)
        out = self.dropout_layer(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out

class StarConcatRpe(nn.Module):
    def __init__(self, config):
        super(StarConcatRpe, self).__init__()
        # Configuration parameters
        self.num_classes = config['num_classes']
        self.dropout_layer = nn.Dropout(config['dropout'])
        self.cnnout = StarConcatCNN(config)
        self.transformerout = RPEAudioTrans(config)

        # Linear Classifier
        self.fc1 = nn.Linear(in_features=128 + config['input_dim'], out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=self.num_classes)

    def forward(self, x):
        x = x.to(device)
        # 为cmdc加的
        # x = x.unsqueeze(1)
        # x = x.permute(0, 1, 3, 2)
        # 拼接两个路径的输出
        cnn_out = self.cnnout(x)
        transformer_out = self.transformerout(x)
        # Fusion and classification
        combined_features = torch.cat((cnn_out, transformer_out), dim=1)
        # 通过两层全连接网络
        out = self.fc1(combined_features)
        out = self.dropout_layer(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out

class AudioGaussianRpe(nn.Module):
    '''
    高斯非线性
    '''
    def __init__(self, config):
        super(AudioGaussianRpe, self).__init__()
        # Configuration parameters
        self.num_classes = config['num_classes']
        self.dropout_layer = nn.Dropout(config['dropout'])
        self.cnnout = GaussianCNN(config)

        self.transformerout = RPEAudioTrans(config)

        # Linear Classifier
        self.fc1 = nn.Linear(in_features=128 + config['input_dim'], out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=self.num_classes)

    def forward(self, x):
        x = x.to(device)
        # 拼接两个路径的输出
        cnn_out = self.cnnout(x)
        transformer_out = self.transformerout(x)
        # Fusion and classification
        combined_features = torch.cat((cnn_out, transformer_out), dim=1)
        # 通过两层全连接网络
        out = self.fc1(combined_features)
        out = self.dropout_layer(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out


class AudioOutProjectRpe(nn.Module):
    '''
    外积
    '''
    def __init__(self, config):
        super(AudioOutProjectRpe, self).__init__()
        # Configuration parameters
        self.num_classes = config['num_classes']
        self.dropout_layer = nn.Dropout(config['dropout'])
        self.cnnout = OuterProductCNN(config)

        self.transformerout = RPEAudioTrans(config)

        # Linear Classifier
        self.fc1 = nn.Linear(in_features=128 + config['input_dim'], out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=self.num_classes)

    def forward(self, x):
        x = x.to(device)
        # 拼接两个路径的输出
        cnn_out = self.cnnout(x)
        transformer_out = self.transformerout(x)
        # Fusion and classification
        combined_features = torch.cat((cnn_out, transformer_out), dim=1)
        # 通过两层全连接网络
        out = self.fc1(combined_features)
        out = self.dropout_layer(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out

