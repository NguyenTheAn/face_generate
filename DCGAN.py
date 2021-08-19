import torch
from torch import nn, optim
from torch.autograd.variable import Variable

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        input_dim = 128
        n_filter = 64
        output_channel = 3

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(input_dim, n_filter * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_filter * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(n_filter * 8, n_filter * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filter * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(n_filter * 4, n_filter * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filter * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(n_filter * 2, n_filter, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filter),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(n_filter, output_channel, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

        self.layer1= nn.Sequential(
            nn.ConvTranspose2d(input_dim, n_filter*8, kernel_size = 4, bias = True),
            nn.ReLU(True),
            # nn.Dropout(0.3)
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(n_filter*8, n_filter*4, kernel_size = 4, stride = 2, padding = 1, bias = True),
            nn.BatchNorm2d(n_filter*4),
            nn.ReLU(True),
            # nn.Dropout(0.3)
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(n_filter*4, n_filter*2, kernel_size = 4, stride = 2, padding = 1, bias = True),
            nn.BatchNorm2d(n_filter*2),
            nn.ReLU(True),
            # nn.Dropout(0.3)
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(n_filter*2, n_filter, kernel_size = 4, stride = 2, padding = 1, bias = True),
            nn.BatchNorm2d(n_filter),
            nn.ReLU(True),
            # nn.Dropout(0.3)
        )

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(n_filter, 32, kernel_size = 4, stride = 2, padding = 1, bias = True),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # nn.Dropout(0.3),
        )

        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(32, output_channel, kernel_size = 4, padding = 1, bias = True),
            # nn.BatchNorm2d(output_channel),
            nn.Tanh()
            # nn.Dropout(0.3),
        )

    def forward(self, x):
        # x = self.main(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        return x
        
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        in_channels = 3
        out_channel = 1
        n_filter = 64

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, n_filter, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(n_filter, n_filter * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filter * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(n_filter * 2, n_filter * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filter * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(n_filter * 4, n_filter * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filter * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(n_filter * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, n_filter, kernel_size = 4, padding=1, stride=2, bias=True),
            nn.BatchNorm2d(n_filter),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.3),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(n_filter, n_filter*2, kernel_size = 4, padding=1, stride=2, bias=True),
            nn.BatchNorm2d(n_filter*2),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.3)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(n_filter*2, n_filter*4, kernel_size = 4, padding=1, stride=2, bias=True),
            nn.BatchNorm2d(n_filter*4),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.3)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(n_filter*4, n_filter*8, kernel_size = 4, padding=1, stride=2, bias=True),
            nn.BatchNorm2d(n_filter*8),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.3),
            nn.Flatten()
        )
        
        self.layer5 = nn.Sequential(
            nn.Linear(8192, 2048, bias=True),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.3),
        )

        self.layer6 = nn.Sequential(
            # nn.Conv2d(n_filter * 8, 1, 4, bias=True),
            nn.Linear(2048, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = self.main(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x.view(-1, 1).squeeze(1)

def ones_target(N):
    return Variable(torch.ones(N))

def zeros_target(N):
    return Variable(torch.zeros(N))

def noise(N):
    return Variable(torch.randn(N, 128, 1, 1))