import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.autograd import Variable
import time
from tqdm import tqdm

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32


class CLSTM2(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=8,
                 hidden_size1=16, hidden_size2=12, hidden_size3=8,
                 num_classes1=5, num_classes2=2, num_classes3=6,
                 num_layers=2, batch_size=20):
        super(CLSTM2, self).__init__()
        self.num_channels = num_channels
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_classes1 = num_classes1  # 연령
        self.num_classes2 = num_classes2  # 성별
        self.num_classes3 = num_classes3  # 지역

        self.conv1 = nn.Conv2d(self.num_channels, self.channel_1, kernel_size=(5, 5), padding=2, bias=True)
        self.conv2 = nn.Conv2d(self.channel_1, self.channel_2, kernel_size=(3, 3), padding=1, bias=True)
        self.conv3 = nn.Conv2d(self.channel_2, 1, kernel_size=(3, 3), padding=1, bias=True)

        self.bn1 = nn.BatchNorm2d(self.channel_1)
        self.bn2 = nn.BatchNorm2d(self.channel_2)
        self.bn3 = nn.BatchNorm2d(1)
        
        self.pool = nn.MaxPool2d(2)

        self.LSTM1 = nn.LSTM(7, self.hidden_size1, self.num_layers, batch_first=True, dropout=0.2)
        self.LSTM2 = nn.LSTM(7, self.hidden_size2, self.num_layers, batch_first=True, dropout=0.2)
        self.LSTM3 = nn.LSTM(7, self.hidden_size3, self.num_layers, batch_first=True, dropout=0.2)

        self.fc1 = nn.Linear(self.hidden_size1, num_classes2)
        self.fc2 = nn.Linear(self.hidden_size2, num_classes1)
        self.fc3 = nn.Linear(self.hidden_size3, num_classes3)

        self.h01 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size1))
        self.c01 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size1))

        self.h02 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size2))
        self.c02 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size2))

        self.h03 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size3))
        self.c03 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size3))

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x):
        ###### 음성 데이터 Feature 추출 ######

        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        
        x = self.pool(x)

        x = x.view(self.batch_size, -1, 200).transpose(2, 1)  # LSTM Input 완성

        ###### LSTM 학습 시작 ######

        # 1st Layer --> 연령
        out1, (h_1, c_1) = self.LSTM1(x, (self.h01, self.c01))  # 1st LSTM Hidden Layer
        h_t1 = out1[:, 1, :]
        out_result1 = self.fc1(h_t1)

        # 2nd Layer --> 성별
        out2, (h_2, c_2) = self.LSTM2(x, (self.h02, self.c02))  # 1st LSTM Hidden Layer
        h_t2 = out2[:, 1, :]
        out_result2 = self.fc2(h_t2)

        # 3rd Layer --> 방언
        out3, (h_3, c_3) = self.LSTM3(x, (self.h03, self.c03))  # 1st LSTM Hidden Layer
        h_t3 = out3[:, 1, :]
        out_result3 = self.fc3(h_t3)

        return out_result1, out_result2, out_result3


class CLSTM(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=8,
                 hidden_size1=16, hidden_size2=12, hidden_size3=8,
                 num_classes1=4, num_classes2=2, num_classes3=6,
                 num_layers=2, batch_size=20):
        super(CLSTM, self).__init__()
        self.num_channels = num_channels
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_classes1 = num_classes1  # 연령
        self.num_classes2 = num_classes2  # 성별
        self.num_classes3 = num_classes3  # 지역

        self.conv1 = nn.Conv2d(self.num_channels, self.channel_1, kernel_size=(5, 5), padding=2, bias=True)
        self.conv2 = nn.Conv2d(self.channel_1, self.channel_2, kernel_size=(3, 3), padding=1, bias=True)
        self.conv3 = nn.Conv2d(self.channel_2, 1, kernel_size=(3, 3), padding=1, bias=True)

        self.bn1 = nn.BatchNorm2d(self.channel_1)
        self.bn2 = nn.BatchNorm2d(self.channel_2)
        self.bn3 = nn.BatchNorm2d(1)

        self.LSTM1 = nn.LSTM(14, self.hidden_size1, self.num_layers, batch_first=True, dropout=0.2)
        self.LSTM2 = nn.LSTM(self.hidden_size1, self.hidden_size2, self.num_layers, batch_first=True, dropout=0.2)
        self.LSTM3 = nn.LSTM(self.hidden_size2, self.hidden_size3, self.num_layers, batch_first=True, dropout=0.2)

        self.fc1 = nn.Linear(self.hidden_size1, num_classes1)
        self.fc2 = nn.Linear(self.hidden_size2, num_classes2)
        self.fc3 = nn.Linear(self.hidden_size3, num_classes3)

        self.h01 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size1))
        self.c01 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size1))

        self.h02 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size2))
        self.c02 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size2))

        self.h03 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size3))
        self.c03 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size3))

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        ###### 음성 데이터 Feature 추출 ######

        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.conv3(x)
        x = F.relu(self.bn3(x))

        x = x.view(self.batch_size, -1, 400).transpose(2, 1)  # LSTM Input 완성

        ###### LSTM 학습 시작 ######

        # 1st Layer --> 연령
        out1, (h_1, c_1) = self.LSTM1(x, (self.h01, self.c01))  # 1st LSTM Hidden Layer
        h_t1 = out1[:, 1, :]
        out_result1 = self.fc1(h_t1)

        # 2nd Layer --> 성별
        out2, (h_2, c_2) = self.LSTM2(out1, (self.h02, self.c02))  # 1st LSTM Hidden Layer
        h_t2 = out2[:, 1, :]
        out_result2 = self.fc2(h_t2)

        # 3rd Layer --> 방언
        out3, (h_3, c_3) = self.LSTM3(out2, (self.h03, self.c03))  # 1st LSTM Hidden Layer
        h_t3 = out3[:, 1, :]
        out_result3 = self.fc3(h_t3)

        return out_result1, out_result2, out_result3


class CLSTM3(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=8,
                 hidden_size1=8, hidden_size2=16, hidden_size3=32,
                 num_classes1=4, num_classes2=2, num_classes3=6,
                 num_layers=2, batch_size=20):
        super(CLSTM3, self).__init__()
        self.num_channels = num_channels
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_classes1 = num_classes1  # 연령
        self.num_classes2 = num_classes2  # 성별
        self.num_classes3 = num_classes3  # 지역

        self.conv1 = nn.Conv2d(self.num_channels, self.channel_1, kernel_size=(5, 5), padding=2, bias=True)
        self.conv2 = nn.Conv2d(self.channel_1, self.channel_2, kernel_size=(3, 3), padding=1, bias=True)
        self.conv3 = nn.Conv2d(self.channel_2, 1, kernel_size=(3, 3), padding=1, bias=True)

        self.bn1 = nn.BatchNorm2d(self.channel_1)
        self.bn2 = nn.BatchNorm2d(self.channel_2)
        self.bn3 = nn.BatchNorm2d(1)

        self.LSTM1 = nn.LSTM(14, self.hidden_size1, self.num_layers, batch_first=True)
        self.LSTM2 = nn.LSTM(self.hidden_size1, self.hidden_size2, self.num_layers, batch_first=True)
        self.LSTM3 = nn.LSTM(self.hidden_size2, self.hidden_size3, self.num_layers, batch_first=True)

        self.fc1 = nn.Linear(400, num_classes1)
        self.fc2 = nn.Linear(400, num_classes2)
        self.fc3 = nn.Linear(400, num_classes3)

        self.h01 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size1))
        self.c01 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size1))

        self.h02 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size2))
        self.c02 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size2))

        self.h03 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size3))
        self.c03 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size3))

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        ###### 음성 데이터 Feature 추출 ######

        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.conv3(x)
        x = F.relu(self.bn3(x))

        x = x.view(self.batch_size, -1, 400).transpose(2, 1)  # LSTM Input 완성

        ###### LSTM 학습 시작 ######

        # 1st Layer --> 연령
        out1, (h_1, c_1) = self.LSTM1(x, (self.h01, self.c01))  # 1st LSTM Hidden Layer
        h_t1 = torch.mean(out1.view(out1.size(0), out1.size(1), -1), dim=2)
        out_result1 = self.fc1(h_t1)

        # 2nd Layer --> 성별
        out2, (h_2, c_2) = self.LSTM2(out1, (self.h02, self.c02))  # 1st LSTM Hidden Layer
        h_t2 = torch.mean(out2.view(out2.size(0), out2.size(1), -1), dim=2)
        out_result2 = self.fc2(h_t2)

        # 3rd Layer --> 방언
        out3, (h_3, c_3) = self.LSTM3(out2, (self.h03, self.c03))  # 1st LSTM Hidden Layer
        h_t3 = torch.mean(out3.view(out3.size(0), out3.size(1), -1), dim=2)
        out_result3 = self.fc3(h_t3)

        return out_result1, out_result2, out_result3


class CLSTM4(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=8,
                 hidden_size1=16, hidden_size2=12, hidden_size3=8,
                 num_classes1=6, num_classes2=5, num_classes3=2,
                 num_layers=2, batch_size=20):
        super(CLSTM4, self).__init__()
        self.num_channels = num_channels
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_classes1 = num_classes1  # 연령
        self.num_classes2 = num_classes2  # 성별
        self.num_classes3 = num_classes3  # 지역

        self.conv1 = nn.Conv2d(self.num_channels, self.channel_1, kernel_size=(5, 5), padding=2, bias=True)
        self.conv2 = nn.Conv2d(self.channel_1, self.channel_2, kernel_size=(3, 3), padding=1, bias=True)
        self.conv3 = nn.Conv2d(self.channel_2, 1, kernel_size=(3, 3), padding=1, bias=True)

        self.bn1 = nn.BatchNorm2d(self.channel_1)
        self.bn2 = nn.BatchNorm2d(self.channel_2)
        self.bn3 = nn.BatchNorm2d(1)

        self.LSTM1 = nn.LSTM(14, self.hidden_size1, self.num_layers, batch_first=True, dropout=0.3)
        self.LSTM2 = nn.LSTM(self.hidden_size1, self.hidden_size2, self.num_layers, batch_first=True, dropout=0.3)
        self.LSTM3 = nn.LSTM(self.hidden_size2, self.hidden_size3, self.num_layers, batch_first=True, dropout=0.3)

        self.fc1 = nn.Linear(400, num_classes1)
        self.fc2 = nn.Linear(400, num_classes2)
        self.fc3 = nn.Linear(400, num_classes3)

        self.fc11 = nn.Linear(self.hidden_size1, num_classes1)
        self.fc21 = nn.Linear(self.hidden_size2, num_classes2)
        self.fc31 = nn.Linear(self.hidden_size3, num_classes3)

        self.h01 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size1))
        self.c01 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size1))

        self.h02 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size2))
        self.c02 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size2))

        self.h03 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size3))
        self.c03 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size3))

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.kaiming_normal_(self.fc11.weight)
        nn.init.kaiming_normal_(self.fc21.weight)
        nn.init.kaiming_normal_(self.fc31.weight)

    def forward(self, x):
        ###### 음성 데이터 Feature 추출 ######

        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.conv3(x)
        x = F.relu(self.bn3(x))

        x = x.view(self.batch_size, -1, 400).transpose(2, 1)  # LSTM Input 완성

        ###### LSTM 학습 시작 ######

        # 1st Layer --> 연령
        out1, (h_1, c_1) = self.LSTM1(x, (self.h01, self.c01))  # 1st LSTM Hidden Layer
        # h_t1 = torch.mean(out1.view(out1.size(0), out1.size(1), -1), dim=2)
        # out_result1 = self.fc1(h_t1)
        h_t1 = out1[:, -1, :]
        out_result1 = self.fc11(h_t1)

        # 2nd Layer --> 성별
        out2, (h_2, c_2) = self.LSTM2(out1, (self.h02, self.c02))  # 1st LSTM Hidden Layer
        # h_t2 = torch.mean(out2.view(out2.size(0), out2.size(1), -1), dim=2)
        # out_result2 = self.fc2(h_t2)
        h_t2 = out2[:, -1, :]
        out_result2 = self.fc21(h_t2)

        # 3rd Layer --> 방언
        out3, (h_3, c_3) = self.LSTM3(out2, (self.h03, self.c03))  # 1st LSTM Hidden Layer
        # h_t3 = torch.mean(out3.view(out3.size(0), out3.size(1), -1), dim=2)
        # out_result3 = self.fc3(h_t3)
        h_t3 = out3[:, -1, :]
        out_result3 = self.fc31(h_t3)

        return out_result2, out_result3, out_result1


class Shared_CNN(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=32, num_classes1=4, num_classes2=2, num_classes3=6):
        super(Shared_CNN, self).__init__()
        self.num_channels = num_channels
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2
        self.num_classes3 = num_classes3

        self.conv1 = nn.Conv2d(self.num_channels, self.channel_1, kernel_size=(3, 3), stride=1)
        self.conv2 = nn.Conv2d(self.channel_1, self.channel_2, kernel_size=(5, 5), padding=(2, 2))

        self.pool = nn.MaxPool2d(2)

        self.dropout = nn.Dropout(0.5)

        self.bn1 = nn.BatchNorm2d(channel_1)
        self.bn2 = nn.BatchNorm2d(channel_2)

        # self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9504, 1024, bias=True)
        self.fc2 = nn.Linear(9504, 1024, bias=True)
        self.fc3 = nn.Linear(9504, 1024, bias=True)

        self.fc10 = nn.Linear(1024, 128, bias=True)
        self.fc20 = nn.Linear(1024, 128, bias=True)
        self.fc30 = nn.Linear(1024, 128, bias=True)

        self.fc11 = nn.Linear(128, self.num_classes1, bias=True)
        self.fc21 = nn.Linear(128, self.num_classes2, bias=True)
        self.fc31 = nn.Linear(128, self.num_classes3, bias=True)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc10.weight)
        nn.init.xavier_uniform_(self.fc20.weight)
        nn.init.xavier_uniform_(self.fc30.weight)
        nn.init.xavier_uniform_(self.fc11.weight)
        nn.init.xavier_uniform_(self.fc21.weight)
        nn.init.xavier_uniform_(self.fc31.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool(x)

        x = self.flatten(x)
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x))
        x3 = F.relu(self.fc3(x))

        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        x3 = self.dropout(x3)

        x1 = F.relu(self.fc10(x1))
        x2 = F.relu(self.fc20(x2))
        x3 = F.relu(self.fc30(x3))

        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        x3 = self.dropout(x3)

        score1 = self.fc11(x1)
        score2 = self.fc21(x2)
        score3 = self.fc31(x3)

        return score1, score2, score3


class Shared_CNN2(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=32, channel_3=64, num_classes1=5, num_classes2=2,
                 num_classes3=6):
        super(Shared_CNN2, self).__init__()
        self.num_channels = num_channels
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.channel_3 = channel_3
        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2
        self.num_classes3 = num_classes3

        self.conv1 = nn.Conv2d(self.num_channels, self.channel_1, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.conv2 = nn.Conv2d(self.channel_1, self.channel_2, kernel_size=(5, 5), padding=(2, 2))

        self.conv_age = nn.Conv2d(self.channel_2, self.channel_3, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.conv_gender = nn.Conv2d(self.channel_2, self.channel_3, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.conv_dialect = nn.Conv2d(self.channel_2, self.channel_3, kernel_size=(3, 3), padding=(1, 1), stride=1)

        self.pool = nn.MaxPool2d(2)

        self.dropout = nn.Dropout(0.3)

        self.bn1 = nn.BatchNorm2d(self.channel_1)
        self.bn2 = nn.BatchNorm2d(self.channel_2)
        self.bn3 = nn.BatchNorm2d(self.channel_3)

        # self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()

        self.fc_1 = nn.Linear(self.channel_3, self.num_classes1, bias=True)
        self.fc_2 = nn.Linear(self.channel_3, self.num_classes2, bias=True)
        self.fc_3 = nn.Linear(self.channel_3, self.num_classes3, bias=True)

        self.fc1 = nn.Linear(6208, 100, bias=True)
        self.fc2 = nn.Linear(6208, 100, bias=True)
        self.fc3 = nn.Linear(6208, 100, bias=True)

        self.fc11 = nn.Linear(100, self.num_classes1, bias=True)
        self.fc21 = nn.Linear(100, self.num_classes2, bias=True)
        self.fc31 = nn.Linear(100, self.num_classes3, bias=True)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv_age.weight)
        nn.init.kaiming_normal_(self.conv_gender.weight)
        nn.init.kaiming_normal_(self.conv_dialect.weight)
        nn.init.kaiming_normal_(self.fc_1.weight)
        nn.init.kaiming_normal_(self.fc_2.weight)
        nn.init.kaiming_normal_(self.fc_3.weight)
        nn.init.kaiming_normal_(self.fc11.weight)
        nn.init.kaiming_normal_(self.fc21.weight)
        nn.init.kaiming_normal_(self.fc31.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool(x)

        x1 = self.conv_age(x)
        x1 = F.relu(self.bn3(x1))

        x2 = self.conv_gender(x)
        x2 = F.relu(self.bn3(x2))

        x3 = self.conv_dialect(x)
        x3 = F.relu(self.bn3(x3))

        x1 = torch.mean(x1.view(x1.size(0), x1.size(1), -1), dim=2)
        score1 = self.fc_1(x1)

        x2 = torch.mean(x2.view(x2.size(0), x2.size(1), -1), dim=2)
        score2 = self.fc_2(x2)

        x3 = torch.mean(x3.view(x3.size(0), x3.size(1), -1), dim=2)
        score3 = self.fc_3(x3)

        """
        x1 = self.flatten(x1)
        x2 = self.flatten(x2)
        x3 = self.flatten(x3)

        x1 = F.relu(self.fc1(x1))
        x2 = F.relu(self.fc2(x2))
        x3 = F.relu(self.fc3(x3))

        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        x3 = self.dropout(x3)

        score1 = self.fc11(x1)
        score2 = self.fc21(x2)
        score3 = self.fc31(x3)
        """

        return score1, score2, score3

class CLSTM_Fin(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=8,
                 hidden_size1=8, hidden_size2=16, hidden_size3=32,
                 num_classes1=4, num_classes2=2, num_classes3=6,
                 num_layers=2, batch_size=20):
        super(CLSTM_Fin, self).__init__()
        self.num_channels = num_channels
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_classes1 = num_classes1  # 연령
        self.num_classes2 = num_classes2  # 성별
        self.num_classes3 = num_classes3  # 지역

        self.conv1 = nn.Conv2d(self.num_channels, self.channel_1, kernel_size=(5, 5), padding=2, bias=True)
        self.conv2 = nn.Conv2d(self.channel_1, self.channel_2, kernel_size=(3, 3), padding=1, bias=True)
        self.conv3 = nn.Conv2d(self.channel_2, 1, kernel_size=(3, 3), padding=1, bias=True)

        self.bn1 = nn.BatchNorm2d(self.channel_1)
        self.bn2 = nn.BatchNorm2d(self.channel_2)
        self.bn3 = nn.BatchNorm2d(1)

        self.LSTM1 = nn.LSTM(14, self.hidden_size1, self.num_layers, batch_first=True)
        self.LSTM2 = nn.LSTM(self.hidden_size1, self.hidden_size2, self.num_layers, batch_first=True)
        self.LSTM3 = nn.LSTM(self.hidden_size2, self.hidden_size3, self.num_layers, batch_first=True)

        self.fc1 = nn.Linear(400, num_classes1)
        self.fc2 = nn.Linear(400, num_classes2)
        self.fc3 = nn.Linear(400, num_classes3)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        ###### 음성 데이터 Feature 추출 ######
        self.h01 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size1, device=x.device))
        self.c01 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size1, device=x.device))

        self.h02 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size2, device=x.device))
        self.c02 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size2, device=x.device))

        self.h03 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size3, device=x.device))
        self.c03 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size3, device=x.device))

        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.conv3(x)
        x = F.relu(self.bn3(x))

        x = x.view(x.size(0), -1, 400).transpose(2, 1)  # LSTM Input 완성

        ###### LSTM 학습 시작 ######

        # 1st Layer --> 연령
        out1, (h_1, c_1) = self.LSTM1(x, (self.h01, self.c01))  # 1st LSTM Hidden Layer
        h_t1 = torch.mean(out1.view(out1.size(0), out1.size(1), -1), dim=2)
        out_result1 = self.fc1(h_t1)

        # 2nd Layer --> 성별
        out2, (h_2, c_2) = self.LSTM2(out1, (self.h02, self.c02))  # 1st LSTM Hidden Layer
        h_t2 = torch.mean(out2.view(out2.size(0), out2.size(1), -1), dim=2)
        out_result2 = self.fc2(h_t2)

        # 3rd Layer --> 방언
        out3, (h_3, c_3) = self.LSTM3(out2, (self.h03, self.c03))  # 1st LSTM Hidden Layer
        h_t3 = torch.mean(out3.view(out3.size(0), out3.size(1), -1), dim=2)
        out_result3 = self.fc3(h_t3)

        return out_result1, out_result2, out_result3

def check_accuracy(loader, model, dataset='valid', batch_size=128):
    batch_size = batch_size
    dataset = dataset
    if dataset == 'train':
        print('\nChecking accuracy on train set')
    elif dataset == 'valid':
        print('\nChecking accuracy on validation set')
    elif dataset == 'test':
        print('Checking accuracy on test set')
    num_correct_age = 0
    num_correct_gender = 0
    num_correct_dialect = 0

    ac_age = []
    ac_gender = []
    ac_dialect = []

    age_class_ac = []
    gender_class_ac = []
    dialect_class_ac = []

    num_samples = 0
    model.eval()

    classes_age = ['청소년', '청년', '중장년', '노년']
    classes_gender = ['여성', '남성']
    classes_dialect = ['수도권', '전라도', '경상도', '충청도', '강원도', '제주도']
    correct_age = list(0. for i in range(len(classes_age)))
    correct_gender = list(0. for i in range(len(classes_gender)))
    correct_dialect = list(0. for i in range(len(classes_dialect)))
    total_age = list(0. for i in range(len(classes_age)))
    total_gender = list(0. for i in range(len(classes_gender)))
    total_dialect = list(0. for i in range(len(classes_dialect)))
    
    duration = []
    
    with torch.no_grad():
        for dict in loader:
            
            x = dict['input']
            age = dict['label'][0]
            gender = dict['label'][1]
            dialect = dict['label'][2]

            x = x.to(device=device, dtype=dtype)
            age = age.to(device=device, dtype=torch.long)
            gender = gender.to(device=device, dtype=torch.long)
            dialect = dialect.to(device=device, dtype=torch.long)

            age_score, gender_score, dialect_score = model(x)
            pred_start = time.perf_counter()
            
            _, preds_age = age_score.max(1)
            _, preds_gender = gender_score.max(1)
            _, preds_dialect = dialect_score.max(1)
            
            pred_end = time.perf_counter()
            
            duration.append(pred_end - pred_start)

            num_correct_age += (preds_age == age).sum()
            num_correct_gender += (preds_gender == gender).sum()
            num_correct_dialect += (preds_dialect == dialect).sum()

            num_samples += preds_age.size(0)

            correct_num_age = (preds_age == age).squeeze()
            correct_num_gender = (preds_gender == gender).squeeze()
            correct_num_dialect = (preds_dialect == dialect).squeeze()

            for i in range(len(dict['label'][0])):
                label = age[i]
                correct_age[label] += correct_num_age[i].item()
                total_age[label] += 1

            for i in range(len(dict['label'][0])):
                label = gender[i]
                correct_gender[label] += correct_num_gender[i].item()
                total_gender[label] += 1

            for i in range(len(dict['label'][0])):
                label = dialect[i]
                correct_dialect[label] += correct_num_dialect[i].item()
                total_dialect[label] += 1

        acc_age = float(num_correct_age) / num_samples
        acc_gender = float(num_correct_gender) / num_samples
        acc_dialect = float(num_correct_dialect) / num_samples

        ac_age.append(acc_age)
        ac_gender.append(acc_gender)
        ac_dialect.append(acc_dialect)

        print('Age     : Got %d / %d correct (%.2f)' % (num_correct_age, num_samples, 100 * acc_age))
        for i in range(len(classes_age)):
            age_class_ac.append(100 * correct_age[i] / total_age[i])
            print("Accuracy of %4s : %2d %%" % (classes_age[i], 100 * correct_age[i] / total_age[i]))

        print('Gender  : Got %d / %d correct (%.2f)' % (num_correct_gender, num_samples, 100 * acc_gender))
        for i in range(len(classes_gender)):
            gender_class_ac.append(100 * correct_gender[i] / total_gender[i])
            print("Accuracy of %4s : %2d %%" % (classes_gender[i], 100 * correct_gender[i] / total_gender[i]))

        print('Dialect : Got %d / %d correct (%.2f)' % (num_correct_dialect, num_samples, 100 * acc_dialect))
        for i in range(len(classes_dialect)):
            dialect_class_ac.append(100 * correct_dialect[i] / total_dialect[i])
            print("Accuracy of %4s : %2d %%" % (classes_dialect[i], 100 * correct_dialect[i] / total_dialect[i]))
            
        print("\nDuration : %.8f초 (per input)\n" % (np.mean(duration) / batch_size))

        return ac_age, ac_gender, ac_dialect, age_class_ac, gender_class_ac, dialect_class_ac


def tester(model, batch_size, model_type="CLSTM"):
    if model_type == "CNN":
        x = torch.zeros((batch_size, 3, 13, 400), dtype=dtype)  # 126000을 데이터 상황에 맞추어 400으로 바꾸어줌
    elif model_type == "CLSTM":
        x = torch.zeros((batch_size, 3, 14, 400), dtype=dtype)  # 126000을 데이터 상황에 맞추어 400으로 바꾸어줌
    test_model = model
    scores1, scores2, scores3 = test_model(x)
    print(scores1.size())
    print(scores2.size())
    print(scores3.size())


def train(model, loader_train, loader_valid, tb_path, optimizer, scheduler, epochs=1, batch_size=128, print_every=1000, model_name="CLSTM"):
    model = model.to(device=device)
    writer = tb_path
    
    time_log = []

    running_loss = 0.0
    epoch_loss = 0.0
    lr = 0.0

    age_class_train_ac = {'청소년': [], '청년': [], '중장년': [], '노년': []}
    gender_class_train_ac = {'여성': [], '남성': []}
    dialect_class_train_ac = {'수도권': [], '전라도': [], '경상도': [], '충청도': [], '강원도': [], '제주도': []}

    age_class_val_ac = {'청소년': [], '청년': [], '중장년': [], '노년': []}
    gender_class_val_ac = {'여성': [], '남성': []}
    dialect_class_val_ac = {'수도권': [], '전라도': [], '경상도': [], '충청도': [], '강원도': [], '제주도': []}

    acc_age = []
    acc_gender = []
    acc_dialect = []

    acc_age_val = []
    acc_gender_val = []
    acc_dialect_val = []

    loss_train = []
    loss_age = []
    loss_gender = []
    loss_dialect = []

    lr_hist = []

    age_weight = torch.tensor([1, 0.318, 0.266, 0.017], dtype=torch.float32)
    gender_weight = torch.tensor([0.612, 1], dtype=torch.float32)
    dialect_weight = torch.tensor([0.16, 0.84, 0.28, 1, 1, 1], dtype=torch.float32)
    age_weight = age_weight.to(device=device)
    gender_weight = gender_weight.to(device=device)
    dialect_weight = dialect_weight.to(device=device)

    for e in tqdm(range(epochs)):
        model.train()
        
        start = time.perf_counter()
        for t, dict in enumerate(loader_train):
            
            x = dict['input']
            age = dict['label'][0]
            gender = dict['label'][1]
            dialect = dict['label'][2]

            x = x.to(device=device, dtype=dtype)
            age = age.to(device=device, dtype=torch.long)
            gender = gender.to(device=device, dtype=torch.long)
            dialect = dialect.to(device=device, dtype=torch.long)

            age_score, gender_score, dialect_score = model(x)
            crit1 = nn.CrossEntropyLoss(weight=age_weight)
            loss1_age = crit1(age_score, age)
            crit3 = nn.CrossEntropyLoss(weight=gender_weight)
            loss1_gender = crit3(gender_score, gender)
            crit2 = nn.CrossEntropyLoss(weight=dialect_weight)
            loss1_dialect = crit2(dialect_score, dialect)

            # if t == 0:
            #    loss = (loss1_age + loss1_gender + loss1_dialect) / 3
            # else:
            #    d_age = prev_age / (loss1_age - prev_age)
            #    d_gender = prev_gender / (loss1_gender - prev_gender)
            #    d_dialect = prev_dialect / (loss1_dialect - prev_dialect)
            #    d_sum = d_age + d_gender + d_dialect
            #
            #    w1 = d_age / d_sum
            #    w2 = d_gender / d_sum
            #    w3 = d_dialect / d_sum
            #
            loss = loss1_age + loss1_gender + loss1_dialect

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            writer.add_scalar('training loss', running_loss / 10, epochs * len(loader_train) + t)
            running_loss = 0

            #if t % print_every == 0:
            #    print(
            #        'Iteration %d --- Train Loss = %.4f --- Age Loss = %.4f --- Gender Loss = %.4f --- Dialect Loss = %.4f' % (
            #        t + 1, loss.item(),
            #        loss1_age.item(),
            #        loss1_gender.item(),
            #        loss1_dialect.item()))

            # if t % print_every == (t-1):
        end = time.perf_counter()
        lr_hist.append(optimizer.param_groups[0]['lr'])
        #print("Current Learning Rate : ", optimizer.param_groups[0]['lr'])
        scheduler.step(e)        
        time_log.append(end - start)
        #print("Epoch %d finished --- Duration : %.4f초" % (e + 1, (end - start)))
        
        if e % print_every == print_every - 1 or e == 0:
            age, gender, dialect, age_class_ac, gender_class_ac, dialect_class_ac = check_accuracy(loader=loader_train,
                                                                                                   model=model, dataset='train', batch_size=batch_size)

            acc_age.append(age)
            acc_gender.append(gender)
            acc_dialect.append(dialect)

            val_age, val_gender, val_dialect, age_class_val, gender_class_val, dialect_class_val = check_accuracy(loader=loader_valid,
                                                                                                                  model=model, dataset='valid', 
                                                                                                                  batch_size=batch_size)
            acc_age_val.append(val_age)
            acc_gender_val.append(val_gender)
            acc_dialect_val.append(val_dialect)

            for i in range(len(list(age_class_train_ac.keys()))):
                age_class_train_ac[list(age_class_train_ac.keys())[i]].append(age_class_ac[i])
                age_class_val_ac[list(age_class_val_ac.keys())[i]].append(age_class_val[i])
            for i in range(len(list(gender_class_train_ac.keys()))):
                gender_class_train_ac[list(gender_class_train_ac.keys())[i]].append(age_class_ac[i])
                gender_class_val_ac[list(gender_class_val_ac.keys())[i]].append(gender_class_val[i])
            for i in range(len(list(dialect_class_train_ac.keys()))):
                dialect_class_train_ac[list(dialect_class_train_ac.keys())[i]].append(dialect_class_ac[i])
                dialect_class_val_ac[list(dialect_class_val_ac.keys())[i]].append(dialect_class_val[i])

        loss_train.append(loss)
        loss_age.append(loss1_age)
        loss_gender.append(loss1_gender)
        loss_dialect.append(loss1_dialect)
        
    torch.save(model.state_dict(), "/root/FM/" + model_name + ".pt")

    df_age_train = pd.DataFrame(age_class_train_ac)
    df_gender_train = pd.DataFrame(gender_class_train_ac)
    df_dialect_train = pd.DataFrame(dialect_class_train_ac)

    df_age_val = pd.DataFrame(age_class_val_ac)
    df_gender_val = pd.DataFrame(gender_class_val_ac)
    df_dialect_val = pd.DataFrame(dialect_class_val_ac)

    import matplotlib.pyplot as plt

    plt.plot(loss_age)
    plt.plot(loss_gender)
    plt.plot(loss_dialect)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Age', 'Gender', 'Dialect'])
    plt.title('Model loss')
    print(plt.show())

    plt.plot(lr_hist, '-')
    plt.xlabel('epoch')
    plt.ylabel('learning rate')
    plt.title('Learning Rate')
    print(plt.show())

    plt.plot(acc_age, '-')
    plt.plot(acc_age_val, '-')
    plt.xticks(list(range(df_age_train.shape[0])), labels=list(range(0, epochs + 1, print_every))) 
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train', 'Validation'])
    plt.title('Age Accuracy')
    print(plt.show())
    
    plt.plot(acc_gender, '-')
    plt.plot(acc_gender_val, '-')
    plt.xticks(list(range(df_gender_train.shape[0])), labels=list(range(0, epochs + 1, print_every))) 
    plt.xlabel('epoch')
    plt.ylabel('accuracy(%)')
    plt.legend(['Train', 'Validation'])
    plt.title('Gender Accuracy')
    print(plt.show())
    
    plt.plot(acc_dialect, '-')
    plt.plot(acc_dialect_val, '-')
    plt.xticks(list(range(df_dialect_train.shape[0])), labels=list(range(0, epochs + 1, print_every))) 
    plt.xlabel('epoch')
    plt.ylabel('accuracy(%)')
    plt.legend(['Train', 'Validation'])
    plt.title('Dialect Accuracy')
    print(plt.show())
    
    df_age_train.plot(xlabel="epoch", ylabel="accuracy(%)")
    plt.xticks(list(range(df_age_train.shape[0])), labels=list(range(0, epochs + 1, print_every))) 
    plt.legend(['10s', '20s~30s', '40s~50s', '60s~'])
    plt.title("Age Accuracy (Train)")
    print(plt.show())

    df_age_val.plot(xlabel="epoch", ylabel="accuracy(%)")
    plt.xticks(list(range(df_age_train.shape[0])), labels=list(range(0, epochs + 1, print_every))) 
    plt.legend(['10s', '20s~30s', '40s~50s', '60s~'])
    plt.title("Age Accuracy (Validation)")
    print(plt.show())
    
    df_gender_train.plot(xlabel="epoch", ylabel="accuracy(%)")
    plt.xticks(list(range(df_gender_train.shape[0])), labels=list(range(0, epochs + 1, print_every))) 
    plt.legend(['Female', 'Male'])
    plt.title("Gender Accuracy (Train)")
    print(plt.show())

    df_gender_val.plot(xlabel="epoch", ylabel="accuracy(%)")
    plt.xticks(list(range(df_gender_train.shape[0])), labels=list(range(0, epochs + 1, print_every))) 
    plt.legend(['Female', 'Male'])
    plt.title("Gender Accuracy (Validation)")
    print(plt.show())

    df_dialect_train.plot(xlabel="epoch", ylabel="accuracy(%)")
    plt.xticks(list(range(df_dialect_train.shape[0])), labels=list(range(0, epochs + 1, print_every))) 
    plt.legend(['GG', 'JL', 'GS', 'CC', 'GW', 'JJ'])
    plt.title("Dialect Accuracy (Train)")
    print(plt.show())

    df_dialect_val.plot(xlabel="epoch", ylabel="accuracy(%)")
    plt.xticks(list(range(df_dialect_train.shape[0])), labels=list(range(0, epochs + 1, print_every))) 
    plt.legend(['GG', 'JL', 'GS', 'CC', 'GW', 'JJ'])
    plt.title("Dialect Accuracy (Validation)")
    print(plt.show())
    
    print("Mean Duration for Training per epoch = %.4f초\n" % (np.mean(time_log)))

    print('---- FINISHED!! ----')

    

def train_weight(model, loader_train, loader_valid, tb_path, optimizer, scheduler, epochs=1, batch_size=128):
    model = model.to(device=device)
    writer = tb_path

    running_loss = 0.0
    epoch_loss = 0.0
    lr = 0.0

    acc_age = []
    acc_gender = []
    acc_dialect = []

    acc_age_val = []
    acc_gender_val = []
    acc_dialect_val = []

    loss_train = []
    loss_age = []
    loss_gender = []
    loss_dialect = []

    lr_hist = []

    age_weight = torch.tensor([0.5, 1, 0.8, 1.4, 0.8], dtype=torch.float32)
    dialect_weight = torch.tensor([0.7, 1, 1, 1, 1, 1.2], dtype=torch.float32)
    age_weight = age_weight.to(device=device)
    dialect_weight = dialect_weight.to(device=device)

    for e in range(epochs):
        model.train()
        for t, dict in enumerate(loader_train):
            x = dict['input']
            age = dict['label'][0]
            gender = dict['label'][1]
            dialect = dict['label'][2]

            x = x.to(device=device, dtype=dtype)
            age = age.to(device=device, dtype=torch.long)
            gender = gender.to(device=device, dtype=torch.long)
            dialect = dialect.to(device=device, dtype=torch.long)

            age_score, gender_score, dialect_score = model(x)
            crit1 = nn.CrossEntropyLoss(weight=age_weight)
            loss1_age = crit1(age_score, age)
            loss1_gender = F.cross_entropy(gender_score, gender)
            crit2 = nn.CrossEntropyLoss(weight=dialect_weight)
            loss1_dialect = crit2(dialect_score, dialect)

            # if t == 0:
            #    loss = (loss1_age + loss1_gender + loss1_dialect) / 3
            # else:
            #    d_age = prev_age / (loss1_age - prev_age)
            #    d_gender = prev_gender / (loss1_gender - prev_gender)
            #    d_dialect = prev_dialect / (loss1_dialect - prev_dialect)
            #    d_sum = d_age + d_gender + d_dialect
            #
            #    w1 = d_age / d_sum
            #    w2 = d_gender / d_sum
            #    w3 = d_dialect / d_sum
            #
            loss = loss1_age + loss1_gender + loss1_dialect

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            writer.add_scalar('training loss', running_loss / 10, epochs * len(loader_train) + t)
            running_loss = 0

            # if t % print_every == 0:
            print(
                'Iteration %d --- Train Loss = %.4f --- Age Loss = %.4f --- Gender Loss = %.4f --- Dialect Loss = %.4f' % (
                t + 1, loss.item(),
                loss1_age.item(),
                loss1_gender.item(),
                loss1_dialect.item()))

            # if t % print_every == (t-1):
        lr_hist.append(optimizer.param_groups[0]['lr'])
        print("Current Learning Rate : ", optimizer.param_groups[0]['lr'])
        scheduler.step(loss)
        print("Epoch %d finished" % (e + 1))

        age, gender, dialect = check_accuracy(loader=loader_train, model=model, dataset='train', batch_size=batch_size)
        acc_age.append(age)
        acc_gender.append(gender)
        acc_dialect.append(dialect)

        val_age, val_gender, val_dialect = check_accuracy(loader=loader_valid, model=model, dataset='valid',
                                                          batch_size=batch_size)
        acc_age_val.append(val_age)
        acc_gender_val.append(val_gender)
        acc_dialect_val.append(val_dialect)

        loss_train.append(loss)
        loss_age.append(loss1_age)
        loss_gender.append(loss1_gender)
        loss_dialect.append(loss1_dialect)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 20))

    plt.subplot(3, 2, 1)
    plt.plot(loss_age)
    plt.plot(loss_gender)
    plt.plot(loss_dialect)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Age', 'Gender', 'Dialect'])
    plt.title('Model loss')

    plt.subplot(3, 2, 4)
    plt.plot(acc_age, '-')
    plt.plot(acc_age_val, '-')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train', 'Validation'])
    plt.title('Age accuracy')

    plt.subplot(3, 2, 5)
    plt.plot(acc_gender, '-')
    plt.plot(acc_gender_val, '-')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train', 'Validation'])
    plt.title('Gender accuracy')

    plt.subplot(3, 2, 6)
    plt.plot(acc_dialect, '-')
    plt.plot(acc_dialect_val, '-')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train', 'Validation'])
    plt.title('dialect accuracy')

    plt.subplot(3, 2, 3)
    plt.plot(lr_hist, '-')
    plt.xlabel('epoch')
    plt.ylabel('learning rate')
    plt.title('Learning Rate')

    print(plt.show())

    print('---- FINISHED!! ----')


def train2(model, loader_train, loader_valid, tb_path, optimizer, epochs=1, print_every=50):
    model = model.to(device=device)
    writer = tb_path

    running_loss = 0.0
    epoch_loss = 0.0

    acc_age = []
    acc_gender = []
    acc_dialect = []

    acc_age_val = []
    acc_gender_val = []
    acc_dialect_val = []

    loss_train = []
    loss_age = []
    loss_gender = []
    loss_dialect = []

    for e in range(epochs):
        model.train()
        for t, dict in enumerate(loader_train):
            x = dict['input']
            age = dict['label'][0]
            gender = dict['label'][1]
            dialect = dict['label'][2]

            x = x.to(device=device, dtype=dtype)
            age = age.to(device=device, dtype=torch.long)
            gender = gender.to(device=device, dtype=torch.long)
            dialect = dialect.to(device=device, dtype=torch.long)

            age_score, gender_score, dialect_score = model(x)
            loss1_age = F.cross_entropy(age_score, age)
            loss1_gender = F.cross_entropy(gender_score, gender)
            loss1_dialect = F.cross_entropy(dialect_score, dialect)

            loss = loss1_age + loss1_gender + loss1_dialect
            optimizer.zero_grad()

            loss1_age.backward(retain_graph=True)
            loss1_gender.backward(retain_graph=True)
            loss1_dialect.backward()

            optimizer.step()

            running_loss += loss.item()

            writer.add_scalar('training loss', running_loss / 10, epochs * len(loader_train) + t)
            running_loss = 0

            # if t % print_every == 0:
            print('Iteration %d --- Train Loss = %.4f' % (t + 1, loss.item()))

            # if t % print_every == (t-1):
        print("Epoch %d finished" % (e + 1))

        age, gender, dialect = check_accuracy(loader=loader_train, model=model, dataset='train')
        acc_age.append(age)
        acc_gender.append(gender)
        acc_dialect.append(dialect)

        val_age, val_gender, val_dialect = check_accuracy(loader=loader_valid, model=model, dataset='valid')
        acc_age_val.append(val_age)
        acc_gender_val.append(val_gender)
        acc_dialect_val.append(val_dialect)

        loss_train.append(loss)
        loss_age.append(loss1_age)
        loss_gender.append(loss1_gender)
        loss_dialect.append(loss1_dialect)

    import matplotlib.pyplot as plt
    plt.plot(acc_age, '-')
    plt.plot(acc_age_val, '-')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train', 'Validation'])
    plt.title('Age accuracy')
    print(plt.show())

    plt.plot(acc_gender, '-')
    plt.plot(acc_gender_val, '-')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train', 'Validation'])
    plt.title('Gender accuracy')
    print(plt.show())

    plt.plot(acc_dialect, '-')
    plt.plot(acc_dialect_val, '-')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train', 'Validation'])
    plt.title('dialect accuracy')
    print(plt.show())

    plt.plot(loss_train)
    plt.plot(loss_age)
    plt.plot(loss_gender)
    plt.plot(loss_dialect)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Age', 'Gender', 'Dialect'])
    plt.title('Model loss')
    print(plt.show())

    print('---- FINISHED!! ----')
    
############################################################

class CnnLSTM_Gender(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=8, hidden_size=16, num_layers=2, batch_size=20, num_classes=2):
        super(CnnLSTM_Gender, self).__init__()
        self.num_channels = num_channels
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(self.num_channels, self.channel_1, kernel_size=(5, 5), padding=2, bias=True)
        self.conv2 = nn.Conv2d(self.channel_1, self.channel_2, kernel_size=(3, 3), padding=1, bias=True)
        self.conv3 = nn.Conv2d(self.channel_2, 1, kernel_size=(3, 3), padding=1, bias=True)

        self.bn1 = nn.BatchNorm2d(self.channel_1)
        self.bn2 = nn.BatchNorm2d(self.channel_2)
        self.bn3 = nn.BatchNorm2d(1)

        self.LSTM = nn.LSTM(14, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(self.hidden_size, num_classes)
        #self.h0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        #self.c0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        
    def forward(self, x):
        self.h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        self.c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.conv3(x)
        x = F.relu(self.bn3(x))

        x = x.view(x.size(0), -1, 400).transpose(2, 1)

        out, _ = self.LSTM(x, (self.h0.detach(), self.c0.detach()))
        h_t = out[:, -1, :]
        out = self.fc1(h_t)

        return out

class RNN_G(nn.Module):
    def __init__(self, input_size=42, hidden_size=16, num_layers=2, batch_size=20, num_classes=2):
        super(RNN_G, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.RNN = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, num_classes)
        #self.h0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        
        nn.init.kaiming_normal_(self.fc1.weight)

    def forward(self, x):
        self.h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        x = x.view(self.batch_size, -1, 400).transpose(2, 1)
        out, _ = self.RNN(x, self.h0)
        h_t = out[:, -1, :]
        out = self.fc1(h_t)
        return out   
    
def check_accuracy_G(loader, model, dataset='valid', batch_size=128):
    dataset = dataset
    if dataset == 'train':
        print('Checking accuracy on train set')
    elif dataset == 'valid':
        print('Checking accuracy on validation set')
    elif dataset == 'test':
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    
    ac_gender = []
    gender_class_ac = []
    classes_gender = ['여성', '남성']
    correct_gender = list(0. for i in range(len(classes_gender)))
    total_gender = list(0. for i in range(len(classes_gender)))
    duration = []
    
    ac = []
    
    model.eval()
    with torch.no_grad():
        for dict in loader:
            x = dict['input']
            y = dict['label'][1]
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            pred_start = time.perf_counter()
            scores = model(x)
            _, preds = scores.max(1)
            
            pred_end = time.perf_counter()
            duration.append(pred_end - pred_start)
                        
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
            correct_num_gender = (preds == y).squeeze()
            
            for i in range(len(dict['label'][1])):
                label = y[i]
                correct_gender[label] += correct_num_gender[i].item()
                total_gender[label] += 1
            
        acc = float(num_correct) / num_samples
        
        ac.append(acc)
        
        print('Gender  : Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        for i in range(len(classes_gender)):
            gender_class_ac.append(100 * correct_gender[i] / total_gender[i])
            print("Accuracy of %4s : %2d %%" % (classes_gender[i], 100 * correct_gender[i] / total_gender[i]))
            
        print("\nDuration : %.6f초 (per input)\n" % (np.mean(duration) / batch_size))
        
        return ac, gender_class_ac

def train_G(model, loader_train, loader_valid, tb_path, optimizer, epochs=1, batch_size=512, print_every=500):
    model = model.to(device=device)
    writer = tb_path
    
    running_loss = 0.0
    
    acc = []
    acc_val = []
    loss_train = []
    
    time_log = []
    gender_class_train_ac = {'여성': [], '남성': []}
    gender_class_val_ac = {'여성': [], '남성': []}
    
    epoch_loss = 0.0
    
    gender_weight = torch.tensor([0.618, 1], dtype=torch.float32)
    gender_weight = gender_weight.to(device=device)

    for e in range(epochs):
        model.train()
        start = time.perf_counter()
        for t, dict in enumerate(loader_train):
            
            x = dict['input']
            y = dict['label'][1]
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.int64)

            scores = model(x)
            crit = nn.CrossEntropyLoss(weight=gender_weight)
            loss = crit(scores, y)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            writer.add_scalar('training loss', running_loss / 10, epochs * len(loader_train) + t)
            running_loss = 0
            
            if t % print_every == 0:
                print('Iteration %d --- Train Loss = %.4f' % (t+1, loss.item()))
        end = time.perf_counter()
        print("Epoch %d finished --- Duration : %.4f초" % (e + 1, (end - start)))
        time_log.append(end - start)
        
        acc1, class_train = check_accuracy_G(loader=loader_train, model=model, dataset='train', batch_size=batch_size)
        acc_val1, class_val = check_accuracy_G(loader=loader_valid, model=model, dataset='valid', batch_size=batch_size)
        
        acc.append(acc1)
        acc_val.append(acc_val1)
        
        for i in range(len(list(gender_class_train_ac.keys()))):
            gender_class_train_ac[list(gender_class_train_ac.keys())[i]].append(class_train[i])
            gender_class_val_ac[list(gender_class_val_ac.keys())[i]].append(class_val[i])
        
        loss_train.append(loss.detach().cpu().item())
    
    df_gender_train = pd.DataFrame(gender_class_train_ac)
    df_gender_val = pd.DataFrame(gender_class_val_ac)
    
    import matplotlib.pyplot as plt
    plt.plot(acc,'-')
    plt.plot(acc_val,'-')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train','Validation'])
    plt.title('Accuracy') 
    plt.show()
    
    plt.plot(loss_train)
    plt.ylabel('loss_train')
    plt.xlabel('epoch')
    plt.title('Model loss') 
    plt.show()
    
    df_gender_train.plot(xlabel="epoch", ylabel="accuracy(%)")
    plt.legend(['Female', 'Male'])
    plt.title("Gender Accuracy (Train)")
    plt.show()

    df_gender_val.plot(xlabel="epoch", ylabel="accuracy(%)")
    plt.legend(['Female', 'Male'])
    plt.title("Gender Accuracy (Validation)")
    plt.show()
    
    print("Mean Duration for Training per epoch = %.4f초\n" % (np.mean(time_log)))
   
    print('---- FINISHED!! ----')
#########################################

class CNN_D(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=32, num_classes=6):
        super(CNN_D, self).__init__()
        self.num_channels = num_channels
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(self.num_channels, self.channel_1, kernel_size=(3, 3), stride=1)
        self.conv2 = nn.Conv2d(self.channel_1, self.channel_2, kernel_size=(5, 5), padding=(2, 2))
        self.pool = nn.MaxPool2d(2)

        self.bn1 = nn.BatchNorm2d(channel_1)
        self.bn2 = nn.BatchNorm2d(channel_2)

        # self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9504, 100, bias=True)
        self.fc2 = nn.Linear(100, self.num_classes, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
class RNN_D(nn.Module):
    def __init__(self, input_size=42, hidden_size=16, num_layers=2, batch_size=20, num_classes=6):
        super(RNN_D, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.RNN = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, num_classes)
        #self.h0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        
        nn.init.kaiming_normal_(self.fc1.weight)

    def forward(self, x):
        self.h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        x = x.view(self.batch_size, -1, 400).transpose(2, 1)
        out, _ = self.RNN(x, self.h0)
        h_t = out[:, -1, :]
        out = self.fc1(h_t)
        return out
    
class CnnLSTM_Dialect(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=8, hidden_size=16, num_layers=2, batch_size=20, num_classes=6):
        super(CnnLSTM_Dialect, self).__init__()
        self.num_channels = num_channels
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(self.num_channels, self.channel_1, kernel_size=(5, 5), padding=2, bias=True)
        self.conv2 = nn.Conv2d(self.channel_1, self.channel_2, kernel_size=(3, 3), padding=1, bias=True)
        self.conv3 = nn.Conv2d(self.channel_2, 1, kernel_size=(3, 3), padding=1, bias=True)

        self.bn1 = nn.BatchNorm2d(self.channel_1)
        self.bn2 = nn.BatchNorm2d(self.channel_2)
        self.bn3 = nn.BatchNorm2d(1)

        self.LSTM = nn.LSTM(14, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(self.hidden_size, num_classes)
        #self.h0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        #self.c0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        
    def forward(self, x):
        self.h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device=device)
        self.c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device=device)
        
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.conv3(x)
        x = F.relu(self.bn3(x))

        x = x.view(x.size(0), -1, 400).transpose(2, 1)

        out, _ = self.LSTM(x, (self.h0, self.c0))
        h_t = out[:, -1, :]
        out = self.fc1(h_t)

        return out
    
def check_accuracy_D(loader, model, dataset='valid', batch_size=128):
    dataset = dataset
    if dataset == 'train':
        print('Checking accuracy on train set')
    elif dataset == 'valid':
        print('Checking accuracy on validation set')
    elif dataset == 'test':
        print('Checking accuracy on test set')
    num_correct = 0
    
    ac = []
    
    num_samples = 0
    
    dialect_class_ac = []
    classes_dialect = ['수도권', '전라도', '경상도', '충청도', '강원도', '제주도']
    correct_dialect = list(0. for i in range(len(classes_dialect)))
    total_dialect = list(0. for i in range(len(classes_dialect)))
    duration = []
    
    model.eval()
    with torch.no_grad():
        for dict in loader:
            x = dict['input']
            y = dict['label'][2]
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            pred_start = time.perf_counter()
            _, preds = scores.max(1)
            
            pred_end = time.perf_counter()
            duration.append(pred_end - pred_start)
            correct_num_dialect = (preds == y).squeeze()
            
            for i in range(len(dict['label'][2])):
                label = y[i]
                correct_dialect[label] += correct_num_dialect[i].item()
                total_dialect[label] += 1
                        
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
            
        acc = float(num_correct) / num_samples
        
        ac.append(acc)
        
        print(total_dialect)
        
        print('Dialect : Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        for i in range(len(classes_dialect)):
            dialect_class_ac.append(100 * correct_dialect[i] / total_dialect[i])
            print("Accuracy of %4s : %2d %%" % (classes_dialect[i], 100 * correct_dialect[i] / total_dialect[i]))
            
        print("\nDuration : %.8f초 (per input)\n" % (np.mean(duration) / batch_size))
        
        return ac, dialect_class_ac

def tester(model, batch_size, model_type="CNN"):
    if model_type == "CNN":
        x = torch.zeros((batch_size, 3, 13, 400), dtype=dtype)  # 126000을 데이터 상황에 맞추어 400으로 바꾸어줌
    elif model_type == "RNN":
        x = torch.zeros((batch_size, 400, 39), dtype=dtype)  # 126000을 데이터 상황에 맞추어 400으로 바꾸어줌
    test_model = model
    scores = test_model(x)
    print(scores.size())


def train_D(model, loader_train, loader_valid, tb_path, optimizer, epochs=1, print_every=100, batch_size=1024):
    model = model.to(device=device)
    writer = tb_path

    running_loss = 0.0
    
    acc = []
    acc_val = []
    loss_train = []
    
    time_log = []
    dialect_class_train_ac = {'수도권': [], '전라도': [], '경상도': [], '충청도': [], '강원도': [], '제주도': []}
    dialect_class_val_ac = {'수도권': [], '전라도': [], '경상도': [], '충청도': [], '강원도': [], '제주도': []}
            
    epoch_loss = 0.0
    dialect_weight = torch.tensor([0.143, 0.865, 0.278, 1, 1, 1], dtype=torch.float32)
    dialect_weight = dialect_weight.to(device=device)

    for e in tqdm(range(epochs)):
        model.train()
        start = time.perf_counter()
        for t, dict in enumerate(loader_train):
            
            x = dict['input']
            y = dict['label'][2]
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.int64)

            scores = model(x)
            crit = nn.CrossEntropyLoss()#weight=dialect_weight)
            loss = crit(scores, y)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            writer.add_scalar('training loss', running_loss / 10, epochs * len(loader_train) + t)
            running_loss = 0
            
            #if t % print_every == 0:
            #    print('Iteration %d --- Train Loss = %.4f' % (t+1, loss.item()))

            #if t % print_every == (t-1):
        end = time.perf_counter()
        #print("Epoch %d finished --- Duration : %.4f초" % (e + 1, (end - start)))
        time_log.append(end - start)
        
        if e % print_every == print_every - 1:
            
            acc1, class_train = check_accuracy_D(loader=loader_train, model=model, dataset='train', batch_size=batch_size)
            acc_val1, class_val = check_accuracy_D(loader=loader_valid, model=model, dataset='valid', batch_size=batch_size)
        
            acc.append(acc1)
            acc_val.append(acc_val1)
            
            for i in range(len(list(dialect_class_train_ac.keys()))):
                dialect_class_train_ac[list(dialect_class_train_ac.keys())[i]].append(class_train[i])
                dialect_class_val_ac[list(dialect_class_val_ac.keys())[i]].append(class_val[i])
        
            loss_train.append(loss.detach().cpu().item())
    
        df_dialect_train = pd.DataFrame(dialect_class_train_ac)
        df_dialect_val = pd.DataFrame(dialect_class_val_ac)
    
    import matplotlib.pyplot as plt
    plt.plot(acc,'-')
    plt.plot(acc_val,'-')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train','Validation'])
    plt.title('Accuracy') 
    plt.show()
    
    plt.plot(loss_train)
    plt.ylabel('loss_train')
    plt.xlabel('epoch')
    plt.title('Model loss') 
    plt.show()
    
    df_dialect_train.plot(xlabel="epoch", ylabel="accuracy(%)")
    plt.legend(['GG', 'JL', 'GS', 'CC', 'GW', 'JJ'])
    plt.title("Dialect Accuracy (Train)")
    plt.show()

    df_dialect_val.plot(xlabel="epoch", ylabel="accuracy(%)")
    plt.legend(['GG', 'JL', 'GS', 'CC', 'GW', 'JJ'])
    plt.title("Dialect Accuracy (Validation)")
    plt.show()
    
    print("Mean Duration for Training per epoch = %.4f초\n" % (np.mean(time_log)))
   
    print('---- FINISHED!! ----')
    
    ########################################

class RNN_A(nn.Module):
    def __init__(self, input_size=42, hidden_size=16, num_layers=2, batch_size=20, num_classes=4):
        super(RNN_A, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.RNN = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, num_classes)
        #self.h0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        
        nn.init.kaiming_normal_(self.fc1.weight)

    def forward(self, x):
        self.h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device=device)
        x = x.view(x.size(0), -1, 400).transpose(2, 1)
        out, _ = self.RNN(x, self.h0)
        h_t = out[:, -1, :]
        out = self.fc1(h_t)
        return out
    
class CNN_A(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=32, num_classes=4):
        super(CNN_A, self).__init__()
        self.num_channels = num_channels
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(self.num_channels, self.channel_1, kernel_size=(3, 3), stride=1)
        self.conv2 = nn.Conv2d(self.channel_1, self.channel_2, kernel_size=(5, 5), padding=(2, 2))
        self.pool = nn.MaxPool2d(2)

        self.bn1 = nn.BatchNorm2d(channel_1)
        self.bn2 = nn.BatchNorm2d(channel_2)

        # self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9504, 100, bias=True)
        self.fc2 = nn.Linear(100, self.num_classes, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x    
    
class CnnLSTM_Age(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=8, hidden_size=16, num_layers=2, batch_size=20, num_classes=4):
        super(CnnLSTM_Age, self).__init__()
        self.num_channels = num_channels
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(self.num_channels, self.channel_1, kernel_size=(5, 5), padding=2, bias=True)
        self.conv2 = nn.Conv2d(self.channel_1, self.channel_2, kernel_size=(3, 3), padding=1, bias=True)
        self.conv3 = nn.Conv2d(self.channel_2, 1, kernel_size=(3, 3), padding=1, bias=True)

        self.bn1 = nn.BatchNorm2d(self.channel_1)
        self.bn2 = nn.BatchNorm2d(self.channel_2)
        self.bn3 = nn.BatchNorm2d(1)

        self.LSTM = nn.LSTM(14, self.hidden_size, self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size, num_classes)
        #self.h0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        #self.c0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.fc1.weight)

    def forward(self, x):
        self.h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device=device)
        self.c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device=device)
        
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.conv3(x)
        x = F.relu(self.bn3(x))

        x = x.view(x.size(0), -1, 400).transpose(2, 1)

        out, _ = self.LSTM(x, (self.h0.detach(), self.c0.detach()))
        h_t = out[:, -1, :]
        out = self.fc1(h_t)

        return out    
    
def check_accuracy_A(loader, model, dataset='valid', batch_size=128):
    dataset = dataset
    if dataset == 'train':
        print('Checking accuracy on train set')
    elif dataset == 'valid':
        print('Checking accuracy on validation set')
    elif dataset == 'test':
        print('Checking accuracy on test set')
    num_correct = 0

    ac = []

    num_samples = 0

    age_class_ac = []
    classes_age = ['청소년', '청년', '중장년', '노년']
    correct_age = list(0. for i in range(len(classes_age)))
    total_age = list(0. for i in range(len(classes_age)))
    duration = []

    model.eval()
    with torch.no_grad():
        for dict in loader:
            x = dict['input']
            y = dict['label'][0]
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            pred_start = time.perf_counter()
            _, preds = scores.max(1)

            pred_end = time.perf_counter()
            duration.append(pred_end - pred_start)
            correct_num_age = (preds == y).squeeze()

            for i in range(len(dict['label'][0])):
                label = y[i]
                correct_age[label] += correct_num_age[i].item()
                total_age[label] += 1

            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

        acc = float(num_correct) / num_samples

        ac.append(acc)

        print(total_age)

        print('age : Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        for i in range(len(classes_age)):
            age_class_ac.append(100 * correct_age[i] / total_age[i])
            print("Accuracy of %4s : %2d %%" % (classes_age[i], 100 * correct_age[i] / total_age[i]))

        print("\nDuration : %.8f초 (per input)\n" % (np.mean(duration) / batch_size))

        return ac, age_class_ac


def tester(model, batch_size, model_type="CNN"):
    if model_type == "CNN":
        x = torch.zeros((batch_size, 3, 13, 400), dtype=dtype)  # 126000을 데이터 상황에 맞추어 400으로 바꾸어줌
    elif model_type == "RNN":
        x = torch.zeros((batch_size, 400, 39), dtype=dtype)  # 126000을 데이터 상황에 맞추어 400으로 바꾸어줌
    test_model = model
    scores = test_model(x)
    print(scores.size())


def train_A(model, loader_train, loader_valid, tb_path, optimizer, epochs=1, print_every=100, batch_size=1024):
    model = model.to(device=device)
    writer = tb_path

    running_loss = 0.0

    acc = []
    acc_val = []
    loss_train = []

    time_log = []
    age_class_train_ac = {'청소년': [], '청년': [], '중장년': [], '노년': []}
    age_class_val_ac = {'청소년': [], '청년': [], '중장년': [], '노년': []}

    epoch_loss = 0.0
    age_weight = torch.tensor([1, 0.318, 0.266, 0.017], dtype=torch.float32)
    age_weight = age_weight.to(device=device)
    

    for e in tqdm(range(epochs)):
        model.train()
        start = time.perf_counter()
        for t, dict in enumerate(loader_train):
            x = dict['input']
            y = dict['label'][1]
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.int64)

            scores = model(x)
            crit = nn.CrossEntropyLoss(weight=age_weight)
            loss = crit(scores, y)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            writer.add_scalar('training loss', running_loss / 10, epochs * len(loader_train) + t)
            running_loss = 0

            # if t % print_every == 0:
            #    print('Iteration %d --- Train Loss = %.4f' % (t+1, loss.item()))

            # if t % print_every == (t-1):
        end = time.perf_counter()
        # print("Epoch %d finished --- Duration : %.4f초" % (e + 1, (end - start)))
        time_log.append(end - start)

        if e % print_every == print_every - 1:

            acc1, class_train = check_accuracy_A(loader=loader_train, model=model, dataset='train',
                                                 batch_size=batch_size)
            acc_val1, class_val = check_accuracy_A(loader=loader_valid, model=model, dataset='valid',
                                                   batch_size=batch_size)

            acc.append(acc1)
            acc_val.append(acc_val1)

            for i in range(len(list(age_class_train_ac.keys()))):
                age_class_train_ac[list(age_class_train_ac.keys())[i]].append(class_train[i])
                age_class_val_ac[list(age_class_val_ac.keys())[i]].append(class_val[i])

            loss_train.append(loss.detach().cpu().item())

        df_age_train = pd.DataFrame(age_class_train_ac)
        df_age_val = pd.DataFrame(age_class_val_ac)

    import matplotlib.pyplot as plt
    plt.plot(acc, '-')
    plt.plot(acc_val, '-')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train', 'Validation'])
    plt.title('Accuracy')
    plt.show()

    plt.plot(loss_train)
    plt.ylabel('loss_train')
    plt.xlabel('epoch')
    plt.title('Model loss')
    plt.show()

    df_age_train.plot(xlabel="epoch", ylabel="accuracy(%)")
    plt.legend(['10s', '20s~30s', '40s~50s', '60s~'])
    plt.title("Age Accuracy (Train)")
    plt.show()

    df_age_val.plot(xlabel="epoch", ylabel="accuracy(%)")
    plt.legend(['10s', '20s~30s', '40s~50s', '60s~'])
    plt.title("Age Accuracy (Validation)")
    plt.show()

    print("Mean Duration for Training per epoch = %.4f초\n" % (np.mean(time_log)))

    print('---- FINISHED!! ----')