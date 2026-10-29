import os
import numpy as np
import cv2
from img2graph import granular_balls_generate

def process_and_save_granular_features(image_dir, output_dir, fold_ind, ids, mode, purity=0.9, threshold=10, var_threshold=20):
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    target_image_files = [f for f in image_files if int(f.split('_')[1]) in ids]

    for file in target_image_files:
        image_path = os.path.join(image_dir, file)
        img = cv2.imread(image_path)

        if img is None:
            print(f"Cannot read the image: {image_path}")
            continue

        print(f"Processing the image: {file} (Fold {fold_ind}, {mode})")
        center_array, adj, edge_attr, center_ = granular_balls_generate(img, purity, threshold, var_threshold)

        base_name = os.path.splitext(file)[0]
        save_path = os.path.join(output_dir, base_name + ".npz")
        np.savez_compressed(save_path,
                            center_array=center_array,
                            adj=adj,
                            edge_attr=edge_attr,
                            center_=center_)
        print(f"Saved features: {save_path}")


if __name__ == '__main__':
    image_dir = r'D:\torchtestto\multimodel\17-coarse\semi\data\all-20\images_256'
    base_output_dir = r'D:\torchtestto\multimodel\17-coarse\semi\data\all-20\images_256_npz'
    split_dir = r'D:\torchtestto\multimodel\17-coarse\semi\data\all-20\h5py-5-200'

    for fold_ind in range(1, 6):
        print(f"\nFold {fold_ind}...")

        split_file = os.path.join(split_dir, f'split_ind_fold{fold_ind}.npz')
        if not os.path.exists(split_file):
            print(f"The index file does not exist: {split_file}")
            continue
        data = np.load(split_file)
        train_ids = data['train_ind'].tolist()
        val_ids = data['val_ind'].tolist()
        test_ids = data['test_ind'].tolist()

        for mode, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
            output_dir = os.path.join(base_output_dir, f'fold_{fold_ind}', mode)
            process_and_save_granular_features(image_dir, output_dir, fold_ind, ids, mode)
