from torch import nn
import torch


class Generator(nn.Module):

    """Generator model: In charge of generate 
    real-like images from a noise distribution


    """
    def __init__(self, latent_size, img_size):
        super(Generator, self).__init__()

        # layers to use
        self.model = nn.Sequential(
            nn.Linear(latent_size, 128),  # Nx100
            nn.LeakyReLU(),
            nn.Linear(128,256), # Nx256
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 512), # Nx512
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024), # Nx1024
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, img_size*img_size), # Nx28*28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
    
class MaxOut(nn.Module):
    """ MaxOut activation function:
    piecewise linear activation function
    used in neural networks. 
    It generalizes other activation functions
    by taking the maximum value of a set of linear 
    combinations of input values
    """
    def __init__(self, num_units, num_pieces):
        super(MaxOut, self).__init__()
        self.num_units = num_units
        self.num_pieces = num_pieces
        self.fc = nn.Linear(num_units, num_units * num_pieces)

    def forward(self, x):
        # Reshape the output to separate pieces
        maxout_output = self.fc(x).view(-1, self.num_pieces, self.num_units)
        # Take the maximum value across pieces
        output, _ = torch.max(maxout_output, dim=1)
        return output
    
    
class Discriminator(nn.Module):
    """Generator model: In charge of classify
    images between real and syntetic generated
    by the generator 

    """
    def __init__(self, img_size):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(img_size*img_size, 512), #N x 512
            MaxOut(512, 4),
            nn.Linear(512, 256), # N x 256
            MaxOut(256, 4),
            nn.Linear(256, 1), # N x 1
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x