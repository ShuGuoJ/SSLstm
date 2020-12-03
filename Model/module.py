import torch
from torch import nn


class BaseModule(nn.Module):
    def __init__(self, inp_channel, hz, nc):
        super().__init__()
        self.lstm = nn.LSTM(inp_channel, hz)
        self.classifier = nn.Linear(hz, nc)

    def forward(self, input):
        pass

class SeLstm(BaseModule):
    def forward(self, input):
        # input: [batchsz, bands] => [bands, batchsz, 1]
        input = input.permute((1,0))
        input = input.unsqueeze(2)
        # out: [bands, batchsz, hz]
        # hn: [layers, batchsz, hz]
        # cn: [layers, batchsz, hz]
        out, (hn, cn) = self.lstm(input)
        spectral_feature = out[-1] # [batchsz, hz]
        # spectral_feature = spectral_feature.sequeeze(0)
        logits = self.classifier(spectral_feature)
        return logits

class SaLstm(BaseModule):
    def forward(self, input):
        # input: [batchsz, h, w, 1] or [batchsz, h, w]
        input = input.squeeze(3) if input.ndim == 4 else input  # input: [batchsz, h, w]
        input = input.permute((1, 0, 2))  # input: [h, batchsz, w]
        out, (hn, cn) = self.lstm(input)
        spatial_feature = out[-1]  # [batchsz, hz]
        logits = self.classifier(spatial_feature)
        return logits
