# GNR 638: Mini Project 2

**Authors:**  
- Ashish Prasad (21d180009)  
- Anannay Jain (210110021)  

## Model Architecture

The project uses a simple Autoencoder model implemented using PyTorch. Below is the architecture of the model:

```python
import torch.nn as nn

class SimpleAE(nn.Module):
    def __init__(self):
        super(SimpleAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=5),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

### Training Details
The model was trained with the following configurations:

Loss Function: MSE Loss
Optimizer: Adam with a learning rate of 1e-3
Scheduler: ReduceLROnPlateau
Epochs: 50
Device: CPU


