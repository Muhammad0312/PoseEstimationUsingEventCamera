import cv2
from PIL import Image, ImageSequence
import os
import re


def natural_sort_key(s):
    """Key function for natural sorting of filenames"""
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]

def create_gif(directory, gif_path, fps=10, resize=(320, 240)):
    # Get the list of image file names in the directory
    image_files = [file for file in os.listdir(directory) if file.endswith('.png')]

    # Sort the image file names alphabetically
    image_files.sort(key=natural_sort_key)

    # image_files = image_files[:400]

    print(image_files)

    # Create a list to store the resized images
    resized_images = []

    # Read and resize the images
    for file in image_files:
        image_path = os.path.join(directory, file)
        image = cv2.imread(image_path)
        if resize is not None:
            image = cv2.resize(image, resize)
        resized_images.append(image)

    # Create a new list to store PIL images
    pil_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in resized_images]

    # Save the images as a GIF with optimization
    pil_images[0].save(gif_path, save_all=True, append_images=pil_images[1:], loop=0, optimize=True, duration=int(1000/fps))

# Example usage
image_directory = 'tv_poster/images'
output_gif_path = 'tv_poster/input_images_tv_poster.gif'
create_gif(image_directory, output_gif_path, fps=200, resize=(600, 300))