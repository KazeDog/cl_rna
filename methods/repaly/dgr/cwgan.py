import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as data
BATCHSIZE=256
def to_onehot(file_path):
    # file_path:保存index的npy文件
    pep = np.load(file_path)
    # print(pep)
    pep = torch.LongTensor(pep)
    # print(pep[0])
    one_hot_pep = F.one_hot(pep, 21).reshape(-1, 30, 21)
    return one_hot_pep
def return_index(one_hot_coding):
    # one_hot = one_hot_coding.numpy()
    index = np.argwhere(one_hot_coding == 1)
    return index[:, -1].reshape(-1, 30)

def load_array(data_arrays, batch_size, is_train=True):

    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

class Discriminator(nn.Module):
    def __init__(self, image_channels, features_d, classes, l_dim=20):
        super(Discriminator, self).__init__()

        self.l_dim = l_dim
        self.l_emb = nn.Embedding(classes, self.l_dim)
        self.image_channels = image_channels
        self.features_d = features_d

        self.optimizer = None

        self.linear = nn.Sequential(
            nn.Linear(features_d * 5 + l_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, features_d * 5),
        )
        self.disc = nn.Sequential(
            nn.Conv2d(image_channels, features_d, kernel_size=(10, 1), stride=1, padding=0),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, (10, 1), 1, 0),
            self._block(features_d * 2, features_d * 4, (10, 1), 1, 0),
            self._block(features_d * 4, features_d * 8, (10, 1), 1, 0),
            nn.Conv2d(features_d * 8, 1, kernel_size=5, stride=1, padding=0),
        )

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, label):
        label = self.l_emb(label)
        x = x.reshape(-1, self.features_d * 5)
        x = torch.cat([x,label],1)
        x= self.linear(x)
        x = x.reshape(-1, 1, self.features_d, 5)
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, image_channels, features_g, classes, l_dim=20):
        super(Generator, self).__init__()

        self.l_dim = l_dim
        self.l_emb = nn.Embedding(classes, self.l_dim)

        self.optimizer = None
        self.name = 'G'
        self.label = 'gen'
        self.features_g = features_g

        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1

            self._block(channels_noise + l_dim, features_g * 16, 5, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, (10, 1), 1, 0),  # img: 8x8
            self._block(features_g * 8, features_g * 4, (10, 1), 1, 0),  # img: 16x16
            self._block(features_g * 4, features_g * 2, (10, 1), 1, 0),  # img: 24*24
            self._block(features_g * 2, image_channels, (10, 1), 1, 0),  # img: 28*28
            nn.Tanh(),
        )

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


    def forward(self, x, label):
        label = self.l_emb(label)
        label = label.unsqueeze(2)
        label = label.unsqueeze(3)
        x = torch.cat([x,label],1)
        x = self.net(x)
        return x

    def sample(self, batch_size, allowed_classes, z_dim):
        self.eval()
        class_num = len(allowed_classes)
        size_for_class = int(batch_size / class_num)
        zs = [torch.randn(size_for_class, z_dim, 1, 1).to(self._device()) for i in range(class_num)]
        xs = []
        ls = []
        for i in range(len(zs)):
            label_i = [allowed_classes[i]] * size_for_class
            label_i = torch.tensor(label_i).to(self._device())
            ls.append(label_i)
            with torch.no_grad():
                x = self.forward(zs[i], label_i)
            _, x = torch.max(x.data, 3)
            x = x.reshape(x.shape[0], self.features_g)
            xs.append(x)
        xs = torch.cat(xs)
        ls = torch.cat(ls)
        return xs, ls


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def gradient_penalty(critic,label, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images,label)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(state, filename="celeba_wgan_gp.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, gen, disc):
    print("=> Loading checkpoint")
    gen.load_state_dict(checkpoint['gen'])
    disc.load_state_dict(checkpoint['disc'])

def props_to_onehot(props):
    if isinstance(props, list):
        props = np.array(props)
    props=props.cpu().detach().numpy()
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]))
    b[np.arange(len(a)), a] = 1
    return torch.FloatTensor(b)

def train_a_batch(generator, discriminator, batch_size, x, y, z_dim, device, d_iters=5):

    LAMBDA_GP = 10

    generator.train()
    discriminator.train()

    for _ in range(d_iters):
        noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake = generator(noise, y)

        disc_real = discriminator(x, y).reshape(-1)
        disc_fake = discriminator(fake, y).reshape(-1)

        gp = gradient_penalty(discriminator, y, x, fake, device=device)
        loss_disc = (
                -(torch.mean(disc_real) - torch.mean(disc_fake)) + LAMBDA_GP * gp
        )
        discriminator.zero_grad()
        loss_disc.backward(retain_graph=True)
        discriminator.optimizer.step()

    gen_fake = discriminator(fake, y).reshape(-1)

    loss_gen = -torch.mean(gen_fake)
    generator.zero_grad()
    loss_gen.backward()
    generator.optimizer.step()

    return {
        'loss_disc': loss_disc,
        'loss_gen': loss_gen
    }
