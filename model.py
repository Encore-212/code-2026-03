import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_star_multistream_model import SParallelCNN
from model_transformer import RAudioTrans


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Audio(nn.Module):
    def __init__(self, config):
        super(Audio, self).__init__()
        self.num_classes = config['num_classes']
        self.dropout_layer = nn.Dropout(config['dropout'])
        self.cnnout = SParallelCNN(config)
        self.transformerout = RAudioTrans(config)
        # Linear Classifier
        self.fc1 = nn.Linear(in_features=128 + config['input_dim'], out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=self.num_classes)

    def forward(self, x):
        x = x.to(device)
        cnn_out = self.cnnout(x)
        transformer_out = self.transformerout(x)
        combined_features = torch.cat((cnn_out, transformer_out), dim=1)
        out = self.fc1(combined_features)
        out = self.dropout_layer(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out


