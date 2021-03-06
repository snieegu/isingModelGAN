import numpy as np
import torch
import time
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from torchsummary import summary
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from isingDataset import IsingDataset
from tempfile import TemporaryFile

ising_data = np.load('ising/cfg_x016_b0100.npy')
ising_data = ising_data.astype(np.float32)
ising_data_ready = torch.Tensor(ising_data).unsqueeze(0)
print("sample ising data", ising_data[0])
print("ising data shape", ising_data.shape)
print("sample ising data shape", ising_data[0].shape)

print("sample ising data tensor", ising_data_ready[0])
print("ising data tensor shape", ising_data_ready.shape)
print("sample ising data tensor shape", ising_data_ready[0].shape)

epochs = 10
batch_size = 100
latent_dim = 16
lr = 0.001

savedModel = "isingOne.pth"
savedDataPath = "outIsing/outputData1epoch.npy"
outfile = TemporaryFile()

transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])
train_dataset = IsingDataset(ising_data, transformation)
data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print("Data batch size: ", next(iter(data_loader)).shape)

print("Data batch: ", next(iter(data_loader)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.main = nn.Sequential(

            nn.ConvTranspose1d(in_channels=1, out_channels=128, kernel_size=3, padding=1),
            # nn.ReLU(),

            nn.ConvTranspose1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            # nn.ReLU(),

            nn.ConvTranspose1d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            # nn.ReLU(),

            nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=3, padding=1),
            # nn.ReLU(),

            nn.Tanh(),

        )

    def forward(self, input):
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(

            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Conv1d(in_channels=64, out_channels=1, kernel_size=3, padding=0),

            nn.Sigmoid()

        )

    def forward(self, input):
        output = self.main(input)
        return output


genNN = Generator()
generator = Generator().to(device)
discriminator = Discriminator().to(device)

print("Discriminator summary:")
summary(discriminator, (1, latent_dim))

print("Generator summary:")
summary(generator, (1, latent_dim))

lossFunction = nn.BCELoss()
generatorOptim = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
discriminatorOptim = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(m.bias.data, 0)


def gen_noise(b_size):
    _generatedNoise = torch.randn(b_size, 1, 16, device=device)
    return _generatedNoise


def show_data(image_tensor, size=(1, 16)):
    flatten_data = image_tensor.detach().cpu().view(-1, *size)
    print(flatten_data)


def save_model():
    torch.save(genNN.state_dict(), savedModel)


def save_data(data, size=(1, 16)):
    flatten_data = data.detach().cpu().view(-1, *size)
    print(flatten_data)
    show_real_data = flatten_data.shape
    _outputData = flatten_data.numpy()
    np.save(savedDataPath, _outputData)


generator.apply(weights_init)
discriminator.apply(weights_init)

real_label = 1.
fake_label = 0.

img_list = []
G_losses = []
D_losses = []
iters = 0

start_time = time.time()
display_step = 250
loss_save_step = 500
save_data_step = 500
mean_generator_loss = 0
mean_discriminator_loss = 0

print("Starting Training Loop...")
for epoch in range(epochs):
    for real in tqdm(data_loader):

        show = real.shape
        cur_batch_size = len(real)
        real = real.unsqueeze(1).to(device)
        show_real = real.shape
        # print(show_real, "\n", real)
        ## Update discriminator ##
        discriminatorOptim.zero_grad()
        disc_real_pred = discriminator(real).reshape(-1)
        disc_real_pred_shape = disc_real_pred.shape
        real_label = (torch.ones(cur_batch_size) * 0.9).to(device)
        real_label_size = real_label.shape
        # Get the discriminator's prediction on the real image and
        # calculate the discriminator's loss on real data
        disc_real_loss = lossFunction(disc_real_pred, real_label)
        # generate the random noise
        fake_noise = gen_noise(cur_batch_size)
        fake_noice_length = len(fake_noise)
        fake_noice_shape = fake_noise.shape
        # generate the fake data by passing the random noise to the generator
        fake = generator(fake_noise)
        fake_shape = fake.shape
        # Get the discriminator's prediction on the fake data generated by generator
        disc_fake_pred = discriminator(fake.detach()).reshape(-1)
        fake_label = (torch.ones(cur_batch_size) * 0.1).to(device)
        disc_fake_pred_shape = disc_fake_pred.shape
        # calculate the discriminator's loss on fake data
        disc_fake_loss = lossFunction(disc_fake_pred, fake_label)
        # Calculate the discriminator's loss by
        # accumulating the real and fake loss
        disc_loss = (disc_fake_loss + disc_real_loss)
        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / display_step
        # Update gradients
        disc_loss.backward()
        # Update optimizer
        discriminatorOptim.step()
        ## Update generator ##
        generatorOptim.zero_grad()

        # Get the discriminator's prediction on the fake data

        disc_fake_pred = discriminator(fake).reshape(-1)
        real_label = (torch.ones(cur_batch_size)).to(device)
        disc_fake_pred_shape = disc_fake_pred.shape
        real_label_shape = real_label.shape
        #  Calculate the generator's loss.
        # the generator wants the discriminator to think that the
        # fake data generated by generator are real
        gen_loss = lossFunction(disc_fake_pred, real_label)
        # Backprop through the generator
        # update the gradients and optimizer.
        gen_loss.backward()
        generatorOptim.step()
        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item() / display_step
        ## Visualization code ##
        if iters % display_step == 0 and iters > 0:
            print(
                f"Epoch:{epoch} Step {iters}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            print("Fake Data")
            show_data(fake.sign())
            # show_data(torch.sign(fake))
            print("Real Data")
            show_data(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0

        # if iters % loss_save_step == 0 and iters > 0:
        #     G_losses.append(gen_loss.item())
        #     D_losses.append(disc_loss.item())
        #
        #     plt.figure(figsize=(7, 5))
        #     plt.title("Generator and Discriminator Loss During Training")
        #     plt.plot(G_losses, label="Generator Loos")
        #     plt.plot(D_losses, label="Discriminator Loss")
        #     plt.xlabel("Iterations")
        #     plt.ylabel("Loss")
        #     plt.legend()
        #     plt.show()
        iters += 1
        G_losses.append(gen_loss.item())
        D_losses.append(disc_loss.item())

        if iters % save_data_step == 0 and iters > 0:
            print("data saved")
            save_data(fake.sign())
    # if epoch % 50 == 0:
    #     torch.save(generator, 'Generator_epoch_{}.pth'.format(epoch))
    #     print('Model saved.')

print('Cost Time: {}s'.format(time.time() - start_time))
plt.figure(figsize=(7, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="Generator Loos")
plt.plot(D_losses, label="Discriminator Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

save_model()
