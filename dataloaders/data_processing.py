import os
import glob
import h5py
from PIL import Image
import numpy as np

slice_num = 0
img_dir = r"D:\torchtestto\multimodel\17-coarse\semi\data\all-20\images_256"
mask_dir = r"D:\torchtestto\multimodel\17-coarse\semi\data\all-20\masks_11_256"
out_dir = r"D:\torchtestto\multimodel\17-coarse\semi\data\all-20\data\slices"
os.makedirs(out_dir, exist_ok=True)

img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png') and f.startswith('Patient_')])
for img_file in img_files:
    img_path = os.path.join(img_dir, img_file)
    mask_path = os.path.join(mask_dir, img_file)

    if not os.path.exists(mask_path):
        print(f"Mask not found: {mask_path}")
        continue

    image = np.array(Image.open(img_path)).astype(np.float32)
    mask = np.array(Image.open(mask_path)).astype(np.uint8)

    image = (image - image.min()) / (image.max() - image.min())

    name = os.path.splitext(img_file)[0]
    h5_path = os.path.join(out_dir, f"{name}.h5")

    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('image', data=image, compression='gzip')
        f.create_dataset('label', data=mask, compression='gzip')

    slice_num += 1

print(f"Converted all PNG slices to H5. Total slices: {slice_num}")




