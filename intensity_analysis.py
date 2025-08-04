import os
import glob
from skimage.io import imread, imsave
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import regionprops, label
from skimage.color import label2rgb
from skimage.util import img_as_ubyte
from skimage.draw import line
import numpy as np
import matplotlib.pyplot as plt

# Directory containing the folders
base_dir = os.getcwd()  # Uses current directory, modify if needed

# Iterate over all directories in the base directory
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path):
        # Process each .ome.tif file in the directory
        for file_path in glob.glob(os.path.join(folder_path, '*.ome.tif')):
            stack = imread(file_path)

            # Ensure the stack is grayscale
            if stack.ndim == 4:
                stack = stack.mean(axis=-1)  # Convert RGB to grayscale by averaging channels

            first_frame = stack[0]
            blurred_first_frame = gaussian(first_frame, sigma=5, preserve_range=True)
            thresh = threshold_otsu(blurred_first_frame)
            binary_mask = (blurred_first_frame > thresh).astype(np.uint8)
            background_mask = 1 - binary_mask
            labeled_mask = label(binary_mask)

            # Visualization of the first few frames
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(first_frame, cmap='gray')
            axes[0].set_title('Original First Frame')
            axes[0].axis('off')

            axes[1].imshow(blurred_first_frame, cmap='gray')
            axes[1].set_title('Blurred First Frame')
            axes[1].axis('off')

            axes[2].imshow(binary_mask, cmap='gray')
            axes[2].set_title('Binary Mask of First Frame')
            axes[2].axis('off')

            # Save the figure
            plt.savefig(os.path.join(folder_path, 'comparison_first_frame.png'), dpi=300)
            plt.close()
            
            #labelled first frame
            fig, ax = plt.subplots()
            ax.imshow(first_frame, cmap='gray')
            regions = regionprops(labeled_mask)
            for region in regions:
                y0, x0 = region.centroid
                ax.text(x0, y0, f'{region.label}', color='cyan', fontsize=6, ha='center', va='center')

            scale_bar_length = 100
            x_start = first_frame.shape[1] - scale_bar_length - 30
            y_start = first_frame.shape[0] - 30
            x_end = x_start + scale_bar_length
            y_end = y_start
            rr, cc = line(y_start, x_start, y_end, x_end)
            first_frame[rr, cc] = first_frame.max()
            ax.plot([x_start, x_end], [y_start, y_end], 'w', lw=4)
            ax.text((x_start + x_end) / 2, y_start - 15, '5 µm', color='white', ha='center', fontsize=6)

            ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(folder_path, 'labeled_first_frame_with_scale.tif'), dpi=300)
            plt.close()

            data = {region.label: [] for region in regionprops(labeled_mask)}
            for i, frame in enumerate(stack):
                blurred_background = gaussian(frame * background_mask, sigma=50, preserve_range=True)
                background_intensity = np.mean(blurred_background[background_mask > 0])

                region_props = regionprops(labeled_mask, intensity_image=frame - background_intensity)
                for region in region_props:
                    data[region.label].append((i + 1, region.mean_intensity, region.mean_intensity * region.area))

            with open(os.path.join(folder_path, 'region_intensities_details_by_region.txt'), 'w') as f:
                f.write("Region_ID,Frame,Mean_Intensity_Above_Background,Integrated_Intensity\n")
                for region_id, records in data.items():
                    for record in records:
                        f.write(f"{region_id},{record[0]},{record[1]},{record[2]}\n")

            for region_id, records in data.items():
                frames = [record[0] for record in records]
                mean_intensities = [record[1] for record in records]
                integrated_intensities = [record[2] for record in records]

                plt.figure(figsize=(10, 5))
                plt.plot(frames, mean_intensities, 'r-o', label='Mean Intensity')
                plt.title(f'Region {region_id} - Mean Intensity Over Time')
                plt.xlabel('Frame Number')
                plt.ylabel('Mean Intensity')
                plt.grid(True)
                plt.savefig(os.path.join(folder_path, f'Region_{region_id}_mean_intensity_plot.png'))
                plt.close()

                plt.figure(figsize=(10, 5))
                plt.plot(frames, integrated_intensities, 'b-x', label='Integrated Intensity')
                plt.title(f'Region {region_id} - Integrated Intensity Over Time')
                plt.xlabel('Frame Number')
                plt.ylabel('Integrated Intensity')
                plt.grid(True)
                plt.savefig(os.path.join(folder_path, f'Region_{region_id}_integrated_intensity_plot.png'))
                plt.close()
