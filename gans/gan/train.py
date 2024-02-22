from gan import Generator, Discriminator
import argparse
from torch import nn
import torch
from tqdm import tqdm
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, RandomSampler, Dataset

parser = argparse.ArgumentParser(description="GAN Paper Implementation")
parser.add_argument('-e', '--epochs', type=int, default=20000, help="Number of overall training iterations")
parser.add_argument('-k', type=int, default="1", help="Number of Discriminator Iterations for epoch (See paper for more details)")
parser.add_argument('-d', '--device', default="cpu", choices=['cpu', 'cuda'], help="Device" )
parser.add_argument('-lr', '--lrate', default=0.0002, help="optimizer learning rate")
parser.add_argument('-b1', '--beta1', default=0.5)
parser.add_argument('-b2', '--beta2', default=0.999)
parser.add_argument('-z', '--latent_dim', default=100, type=int, help="Latent dimension")
parser.add_argument('-s', '--img_size', default=28, type=int, help="Image size")
parser.add_argument('-b', '--batch_size', default=64, type = int)
cfg = parser.parse_args()




# Models 
G = Generator(cfg.latent_dim, cfg.img_size)
D = Discriminator(cfg.img_size)
G.to(cfg.device)
D.to(cfg.device)

# Losses
D_LOSS = nn.BCELoss()
G_LOSS = nn.BCELoss()

# Optimizers
d_optimizer = torch.optim.Adam(D.parameters(), lr = cfg.lrate, betas = (cfg.beta1, cfg.beta2))
g_optimizer = torch.optim.Adam(G.parameters(), lr = cfg.lrate, betas = (cfg.beta1, cfg.beta2))

# Datasets (Images and Noise)
transform = transforms.Compose([
    transforms.Resize((cfg.img_size, cfg.img_size)),
    transforms.ToTensor(),  # Convert PIL image or numpy array to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the tensor with mean and standard deviation
])

dataset = MNIST(root = '', download = True, transform =transform )
sampler = RandomSampler(dataset) # To get random images each iteration

# DataLoader
original_dl = DataLoader(dataset, batch_size = cfg.batch_size, sampler = sampler, pin_memory=torch.cuda.is_available())



def train():
    for epoch in tqdm(range(cfg.epochs)):
        ##############################
        ## Discriminator Optimization 
        ##############################
        for k in range(cfg.k):
            x, _ = next(iter(original_dl)) # random batch
            x = x.to(cfg.device)
            batch_size = x.size()[0]
            x_b = x.view(batch_size, -1)
            d_optimizer.zero_grad()

            # Models outputs
            z = torch.randn(batch_size, cfg.latent_dim)
            z = z.to(cfg.device)
            G_z = G(z)
            D_x  = D(x_b)
            D_G_z = D(G_z.detach())


            # Outputs & Images
            samples = torch.cat([D_G_z, D_x]).to(cfg.device)
            targets = torch.cat([torch.zeros(D_G_z.size()[0]), torch.ones(D_x.size()[0])]).to(cfg.device)

            # Loss
            d_loss = D_LOSS(samples, targets.unsqueeze(-1))
            d_loss.backward()

            # Adjust learning weights
            d_optimizer.step()

        ##############################
        ## Generator Optimization 
        ##############################
        g_optimizer.zero_grad()
        
        z = torch.randn(batch_size, cfg.latent_dim).to(cfg.device)
        G_z = G(z)
        D_G_z = D(G_z)
        targets = torch.zeros(D_G_z.size()[0]).to(cfg.device)
        # Loss
        g_loss = -G_LOSS(D_G_z, targets.unsqueeze(-1))
        g_loss.backward()

        # Adjust learning weights
        g_optimizer.step()

        if epoch%1000 == 0:
            print(f"({epoch}/{cfg.epochs}) | D Loss {d_loss:.3f} | G Loss {g_loss:.3f}")
        
    
    torch.save(G.state_dict(), f"G.pt")
    torch.save(D.state_dict(), f"D.pt")



if __name__ == "__main__":
    train()









