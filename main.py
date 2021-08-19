import os
from imutils import paths
from dataloader import Dataset
from DCGAN import *
import torch
import numpy as np
import torchvision.utils as vutils

batch_size = 128
LR = 0.0002

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset = Dataset("../data/img_align_celeba")

trainloader = torch.utils.data.DataLoader(train_dataset,  batch_size=batch_size, shuffle=True, num_workers=4)

gen = Generator()
gen.to(device)
gen.apply(weights_init)
dis = Discriminator()
dis.to(device)
dis.apply(weights_init)
d_optimizer = optim.Adam(dis.parameters(), lr = LR, betas=(0.5, 0.999))
g_optimizer = optim.Adam(gen.parameters(), lr = LR*10, betas=(0.5, 0.999))

loss = nn.MSELoss()


def train(epochs, verbose):
    print("=================training================")
    for epoch in range(epochs):
        # pbar = tqdm(enumerate(trainloader), total=len(trainloader))
        for i, real_data in enumerate(trainloader):
            # pbar.set_description(f"Trainning epoch {epoch}")
            N = real_data.size(0)
            real_data = real_data.to(device)
            fake_data = gen(noise(N).to(device))

            # train discriminator
            d_optimizer.zero_grad()
            real_pred = dis(real_data)
            real_label = ones_target(N).to(device)
            real_error = loss(real_pred, real_label)
            # real_error.backward()

            fake_pred = dis(fake_data.detach())
            fake_label = zeros_target(N).to(device)
            fake_error = loss(fake_pred, fake_label)
            # fake_error.backward()

            d_error = fake_error + real_error
            # d_error = torch.sum((real_pred - 1.0)**2) + torch.sum((fake_pred)**2)
            d_error.backward()
            d_optimizer.step()

            # train generator
            g_optimizer.zero_grad()
            pred = dis(fake_data)

            g_error = loss(pred, real_label)
            g_error.backward()
            g_optimizer.step()

            print(f"Epoch {epoch} step {i}: Generator loss: {g_error} / Discriminator loss: {d_error}\n")
            if i % 100 == 0:
                print('saving the output')
                vutils.save_image(real_data,'results/real_samples.png',normalize=True)
                fake = gen(test_noise)
                vutils.save_image(fake.detach(),'results/fake_samples_epoch_%03d.png' % (epoch),normalize=True)
            

test_noise = noise(128).to(device)

train(200, 10)
            