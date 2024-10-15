import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F

from unet_model import SimpleUnet
from add_noise import AddNoiseProcess
from dataset import load_transformed_dataset


T = 300
BATCH_SIZE = 32

model = SimpleUnet()
device = "cuda:2" if torch.cuda.is_available() else "cpu"
model.to(device)

dataset = load_transformed_dataset()
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = Adam(model.parameters(), lr=0.001)
epochs = 100 # Try more!

add_noise_process = AddNoiseProcess(T)

def get_loss(model, x_0, t,device):
    x_noisy, noise = add_noise_process.forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
      optimizer.zero_grad()

      t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
      loss = get_loss(model, batch[0], t,device)
      loss.backward()
      optimizer.step()
      if step % 100 == 0:
        print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")