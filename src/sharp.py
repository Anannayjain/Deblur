import os
import shutil

shp=os.listdir('C:/Users/HP/Desktop/Deblur/image-deblurring-using-deep-learning/train_sharp/train/train_sharp')

source_directory = 'C:/Users/HP/Desktop/Deblur/image-deblurring-using-deep-learning/train_sharp/train/train_sharp'  # Specify the path to the parent directory containing the folders
destination_directory = 'C:/Users/HP/Desktop/Deblur/image-deblurring-using-deep-learning/input/sharp'  # Specify the destination directory for storing the first images

i=0

for folder in shp:
    # Get the list of files in the current folder
    files_in_folder = os.listdir(os.path.join(source_directory, folder))
    if len(files_in_folder) > 0:
        # Get the path of the first image in the folder
        first_image_path = os.path.join(source_directory, folder, files_in_folder[0])
        
        custom_name= f'{i}.jpg'
        i=i+1
        # Copy the first image to the destination directory
        shutil.copy(first_image_path, os.path.join(destination_directory, custom_name))