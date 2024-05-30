import numpy as np
import matplotlib.pyplot as plt

def plot_slices(mri_slice, seg_slice, plt_title:str , omit_background=True, show=True, save_path=None):
    # Create a masked array where only values 1, 2, and 3 are included
    # and other values are set to be transparent
    mask = np.isin(seg_slice, [1, 2, 3])
    masked_seg_slice = np.where(mask, seg_slice, np.nan)  # Replace 0s with NaN for transparency

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(mri_slice.T, cmap='gray', origin='lower')
    ax[1].imshow(mri_slice.T, cmap='gray', origin='lower')

    # Give Titles to the plots
    ax[0].set_title('MRI Slice')
    ax[1].set_title('Segmentation Slice')

    if omit_background==True:
        # Only overlay areas where mask is True, with the segmentation mask using a colormap
        ax[1].imshow(masked_seg_slice.T, cmap='jet', alpha=0.5, origin='lower')
        
    else:
        ax[1].imshow(seg_slice.T, cmap='jet', alpha = 0.5, origin='lower')      #Overlay with transparency

    if save_path:
        plt.savefig(save_path)  # Save the figure to the specified path

    #plt.title(plt_title)

    if show:
        plt.show()      # Show the plot if requested
    else:
        plt.close(fig)  # Only close the figure if not showing it