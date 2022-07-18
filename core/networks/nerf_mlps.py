import torch
import torch.nn as nn
import torch.nn.functional as F

class DensityMLP(nn.Module):

    def __init__(self, input_ch, output_ch, skips=[4], D=8, W=256,
                 act_fn=nn.ReLU(inplace=True)):
        super().__init__()
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.skips = skips

        self.W = W
        self.D = D

        self.act_fn = act_fn
        self.init_network()
    
    def init_network(self):
        W, D = self.W, self.D

        layers = [nn.Linear(self.input_ch, W)]

        for i in range(D-1):
            if i not in self.skips:
                layers += [nn.Linear(W, W)]
            else:
                layers += [nn.Linear(W + self.input_ch, W)]

        self.pts_linears = nn.ModuleList(layers)
        self.alpha_linear = nn.Linear(W, self.output_ch)

    def forward(self, x):
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = self.act_fn(h)
            if i in self.skips:
                h = torch.cat([density_inputs, h], -1)
        alpha = self.alpha_linear(h)
        return torch.cat([alpha, h], dim=-1)
