# ini cara agar di tulis jadi script yang nanti nya bisa di import

import torch
from torch import nn, optim
from jcopdl.layers import linear_block

class Disciminator (nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            linear_block(784,512, activation = "lrelu"), # activation ini untuk menghindari angka nol
            linear_block(512,256, activation = "lrelu"),
            linear_block(256,128, activation = "lrelu"),
            linear_block(128,1, activation = "sigmoid")
        )
    def forward(self,x):
        return self.fc(x)
    
class Generator (nn.Module):
    def __init__(self, n_latent):
        super().__init__()
        self.n_latent = n_latent
        self.fc = nn.Sequential(
            linear_block(n_latent,128, activation = "lrelu"),
            linear_block(128,256, activation = "lrelu", batch_norm = True), # activation ini untuk menghindari angka nol
            linear_block(256,512, activation = "lrelu", batch_norm = True),
            linear_block(512,1024, activation = "lrelu", batch_norm = True),
            linear_block(1024,784, activation = "tanh") # karna tadi di generator input gambar nya dari -1 sampe 1 maka nya generate fake image harus menyesuaikan juga
        )
    def forward(self,x):
        return self.fc(x)
    
    # buat satu fungsi untuk generate n(seberapa banyak) ukuran latent space atau random number seukuran dengan n_latent
    def generate (self, n, device):
        z= torch.randn((n, self.n_latent), device=device) # generate random number sebanyak n kali latent space kita atau generate n gambar palsu
        return self.fc(z) # dari random number ini kita masukin ke arsitektur generator kita untuk di generate gambar
