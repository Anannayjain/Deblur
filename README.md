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
```

### Training Details
The model was trained with the following configurations:

Loss Function: MSE Loss
Optimizer: Adam with a learning rate of 1e-3
Scheduler: ReduceLROnPlateau
Epochs: 50
Device: CPU

### Code for Adding Noise

```python
for i, img in tqdm(enumerate(images), total=len(images)):
    img = cv2.imread(f"{src_dir}/{images[i]}")
    # add gaussian blurring
    blur = cv2.GaussianBlur(img, (11,11), 1.6)
    cv2.imwrite(f"{dst_dir}/{images[i]}", blur)
print('DONE')
```
## Training Curve

![image](https://github.com/user-attachments/assets/6437cbd7-9170-435d-8134-28ea2748052d)




