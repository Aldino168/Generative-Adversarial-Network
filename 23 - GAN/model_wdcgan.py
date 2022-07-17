import torch
from torch import nn
from jcopdl.layers import conv_block, tconv_block, linear_block

def conv(c_in, c_out, batch_norm=True, activation="lrelu"):
    return conv_block(c_in, c_out, kernel=4, stride=2, pad=1, bias=False, batch_norm=batch_norm, activation=activation, pool_type=None)

def tconv(c_in, c_out, batch_norm=True, activation="lrelu"):
    return tconv_block(c_in, c_out, kernel=4, stride=2, pad=1, bias=False, batch_norm=batch_norm, activation=activation, pool_type=None)  


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            conv(3, 32, batch_norm=False),          
            conv(32, 64),
            conv(64, 128),
            conv(128, 256),
            conv_block(256, 1, kernel=4, stride=1, pad=0, bias=False, activation=None, pool_type=None),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
    def clip_weights(self, vmin=-0.01, vmax=0.01):
        for p in self.parameters():
            p.data.clamp_(vmin, vmax)    


class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.tconv = nn.Sequential(
            tconv_block(z_dim, 512, kernel=4, stride=2, pad=1, bias=False, activation="lrelu", pool_type=None),
            tconv(512, 256),
            tconv(256, 128),
            tconv(128, 64),
            tconv(64, 32),
            tconv(32, 3, activation="tanh", batch_norm=False)
        )
        
    def forward(self, x):
        return self.tconv(x)

    def generate(self, n, device):
        z = torch.randn((n, self.z_dim, 1, 1), device=device)
        return self.tconv(z)
