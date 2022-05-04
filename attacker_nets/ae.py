# '''
# @Description  :
# @Author       : Chi Liu
# @Date         : 2022-03-26 23:17:55
# @LastEditTime : 2022-03-27 02:28:17
# '''
import torch.nn as nn
import torch.nn.functional as F


class AE(nn.Module):

    def __init__(self):
        super(AE, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 8, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 8, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(8, 3, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(3)
        self.convt1 = nn.ConvTranspose2d(8, 8, 2, stride=2)
        self.convt2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.convt3 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        self.convt4 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.pool = nn.MaxPool2d(2, 2)
        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight.data)
                nn.init.normal(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(self.bn1(x)))  # out [16, N/2, N/2, 1]
        x = self.conv2(x)
        x = self.pool(F.relu(self.bn2(x)))  # out [8, N/4, N/4, 1]
        x = self.conv3(x)
        x = self.pool(F.relu(self.bn2(x)))  # out [8, N/8, N/8, 1]

        x = self.convt1(x)
        x = F.relu(self.bn2(x))  # out [8, N/4, N/4, 1]
        x = self.convt2(x)
        x = F.relu(self.bn1(x))  # out [16, N/2, N/2, 1]
        x = self.convt3(x)
        x = F.relu(self.bn3(x))  # out [32, N, N, 1]
        x = self.convt4(x)
        x = F.relu(self.bn1(x))  # out [16, 2N, 2N, 1]

        x = self.conv2(x)
        x = F.relu(self.bn2(x))  # out [8, 2N, 2N, 1]
        x = self.conv4(x)
        x = F.relu(self.bn4(x))  # out [3, 2N, 2N, 1]

        return x
# class AE(nn.Module):
#     def __init__(self, **kwargs):
#         super().__init__()
#         self.encoder_hidden_layer = nn.Linear(
#             in_features=kwargs["input_shape"], out_features=128
#         )
#         self.encoder_output_layer = nn.Linear(
#             in_features=128, out_features=128
#         )
#         self.decoder_hidden_layer = nn.Linear(
#             in_features=128, out_features=128
#         )
#         self.decoder_output_layer = nn.Linear(
#             in_features=128, out_features=kwargs["input_shape"]
#         )

#     def forward(self, features):
#         activation = self.encoder_hidden_layer(features)
#         activation = torch.relu(activation)
#         code = self.encoder_output_layer(activation)
#         code = torch.relu(code)
#         activation = self.decoder_hidden_layer(code)
#         activation = torch.relu(activation)
#         activation = self.decoder_output_layer(activation)
#         reconstructed = torch.relu(activation)
#         return reconstructed
