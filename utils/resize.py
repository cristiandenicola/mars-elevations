from PIL import Image
import numpy as np

def resize_image(image, target_size):
    pil_img = Image.fromarray(image)
    resized_img = pil_img.resize(target_size, Image.BILINEAR)
    return np.array(resized_img)