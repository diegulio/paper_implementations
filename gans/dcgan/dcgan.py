from torch import nn
import torch


class Generator(nn.Module):
    """Generator model: In charge of generate 
    real-like images from a noise distribution


    """
    def __init__(self, latent_size, n_channels):
        super(Generator, self).__init__()

        self.linear_projection = nn.Linear(latent_size, 4*4*512)

        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64, n_channels, 4, 2, 1),
            nn.Tanh()
        )

        self.init_weights()

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                # Mean and standard deviation for the normal distribution
                mean, std = 0.0, 0.2

                # Initialize the weights with a normal distribution
                nn.init.normal_(layer.weight, mean=mean, std=std)

    def forward(self, x):
        x = self.linear_projection(x)
        x = x.view(-1, 512, 4, 4)
        x = self.upsampling(x)
        return x


class Discriminator(nn.Module):
  def __init__(self, n_channels):
    """Generator model: In charge of classify
    images between real and syntetic generated
    by the generator

    """
    super(Discriminator, self).__init__()

    self.model = nn.Sequential(
        nn.Conv2d(n_channels, 64, 4, 2, 1),
        nn.LeakyReLU(0.2, inplace = True),
        nn.Conv2d(64, 128, 4, 2, 1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace = True),
        nn.Conv2d(128, 256, 4, 2, 1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace = True),
        nn.Conv2d(256, 512, 4, 2, 1),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace = True),
        nn.Flatten(),
        nn.Linear(512*4*4, 1),
        nn.Sigmoid()

    )


    self.init_weights()

  def init_weights(self):
    for layer in self.modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            # Mean and standard deviation for the normal distribution
            mean, std = 0.0, 0.2

            # Initialize the weights with a normal distribution
            nn.init.normal_(layer.weight, mean=mean, std=std)


  def forward(self, x):
    x = self.model(x)
    return x

