import numpy as np

def extract_patches(img, patch_size=256, stride=128):
    patches = []
    h, w = img.shape[-2], img.shape[-1]
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = img[..., i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return patches