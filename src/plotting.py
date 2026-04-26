import numpy as np
import matplotlib.pyplot as plt

def plot_patch_importance(importances, patches, img_size=28):
    """Map patch importances back to pixel grid and plot."""
    pixel_importance = np.zeros(img_size * img_size)

    for patch_idx, patch in enumerate(patches):
        pixel_importance[patch] = importances[patch_idx]

    plt.figure(figsize=(5, 4))
    plt.imshow(pixel_importance.reshape(img_size, img_size))
    plt.colorbar(label='LOCO importance')
    plt.title(f'{len(patches)} patches')
    plt.axis('off')
    plt.show()