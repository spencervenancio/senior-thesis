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


def plot_selected(minshap_result, patches, img_size=28):
    """Map MinShap-selected patches (rejected=True) back to pixel grid and plot."""
    rejected = minshap_result['rejected']
    pixel_mask = np.zeros(img_size * img_size)

    for patch_idx, patch in enumerate(patches):
        if rejected[patch_idx]:
            pixel_mask[patch] = 1.0

    n_selected = int(rejected.sum())
    plt.figure(figsize=(5, 4))
    plt.imshow(pixel_mask.reshape(img_size, img_size), cmap='binary', vmin=0, vmax=1)
    plt.title(f'{n_selected} / {len(patches)} patches selected')
    plt.axis('off')
    plt.show()