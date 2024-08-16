import os
from skimage.metrics import peak_signal_noise_ratio
from skimage.io import imread
import torch
from torchvision import transforms
from PIL import Image  
import models

model = models.CNN()
checkpoint = torch.load(r'C:/Users/HP/Desktop/Deblur/image-deblurring-using-deep-learning/outputs/model.pth')
# print(checkpoint.keys())
# model.load_state_dict(checkpoint['odict_keys'])
# Load weights and biases into the model
model.conv1.weight = torch.nn.Parameter(checkpoint['conv1.weight'])
model.conv1.bias = torch.nn.Parameter(checkpoint['conv1.bias'])

model.conv2.weight = torch.nn.Parameter(checkpoint['conv2.weight'])
model.conv2.bias = torch.nn.Parameter(checkpoint['conv2.bias'])

model.conv3.weight = torch.nn.Parameter(checkpoint['conv3.weight'])
model.conv3.bias = torch.nn.Parameter(checkpoint['conv3.bias'])

# Set the model to evaluation mode
model.eval()


transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize((256, 448)),
    transforms.ToTensor(),
])

for path in os.listdir(r'C:/Users/HP/Desktop/Deblur/image-deblurring-using-deep-learning/mp2_test/custom_test/blur'):
    input_image = Image.open(r'C:/Users/HP/Desktop/Deblur/image-deblurring-using-deep-learning/mp2_test/custom_test/blur/' + path)
    input_tensor = transform(input_image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Postprocess output if needed (e.g., convert tensors to images)
    # Assuming output_tensor is the output image tensor
    output_image = transforms.ToPILImage()(output_tensor.squeeze(0))

    # Save the output image to a file
    output_image.save(r'C:/Users/HP/Desktop/Deblur/image-deblurring-using-deep-learning/mp2_test/custom_test/model_sharp/' + path)




def psnr_between_folders(folder1, folder2):
    psnr_values = []
    
    # Get list of filenames in folder1
    filenames = os.listdir(folder1)
    
    for filename in filenames:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Read corresponding images from both folders
            img_path1 = os.path.join(folder1, filename)
            img_path2 = os.path.join(folder2, filename)
            img1 = imread(img_path1)
            img2 = imread(img_path2)
            
            # Compute PSNR between corresponding images
            psnr = peak_signal_noise_ratio(img1, img2)
            psnr_values.append(psnr)
    
    # Compute average PSNR across all images
    avg_psnr = sum(psnr_values) / len(psnr_values)

    print (len(psnr_values))
    
    return avg_psnr

# Example usage:
folder1 = r"C:/Users/HP/Desktop/Deblur/image-deblurring-using-deep-learning/mp2_test/custom_test/sharp"
folder2 = r"C:/Users/HP/Desktop/Deblur/image-deblurring-using-deep-learning/mp2_test/custom_test/model_sharp"

avg_psnr = psnr_between_folders(folder1, folder2)
print(f"Average PSNR between corresponding images: {avg_psnr} dB")
