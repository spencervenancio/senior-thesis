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


def plot_local_selected(minshap_result, patches, x_S, img_size=28):
    """Show the input image with MinShap-selected patches highlighted as an overlay."""
    rejected = minshap_result['rejected']
    image = x_S.reshape(img_size, img_size)

    mask = np.zeros(img_size * img_size)
    for patch_idx, patch in enumerate(patches):
        if rejected[patch_idx]:
            mask[patch] = 1.0
    mask = mask.reshape(img_size, img_size)

    n_selected = int(rejected.sum())
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(image, cmap='gray')
    ax.imshow(mask, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
    ax.set_title(f'{n_selected} / {len(patches)} patches selected')
    ax.axis('off')
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

def plot_saliency(attribution, X, y, idx=0, show_image=True):
    image = X[idx].reshape(28, 28).detach().numpy()
    attr = attribution.squeeze().reshape(28, 28).detach().numpy()
    fig, ax = plt.subplots(figsize=(10, 3))
    if show_image:
        ax.imshow(image, cmap='gray')
    ax.imshow(attr, cmap='hot', alpha=0.5 if show_image else 1.0)
    plt.tight_layout()
    plt.show()
    return fig, ax