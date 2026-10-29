import argparse
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from img2graph import granular_balls_generate, cal_bound
from model import GCN_8_plus

random.seed(0)
torch.manual_seed(0)

def visualize_pixel_matrix(pixel_matrix, title='gray Image'):
    plt.figure(figsize=(6, 3))
    plt.imshow(pixel_matrix, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

colors = np.array([
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [255, 105, 180],
    [128, 0, 0],
    [0, 128, 0],
    [0, 0, 128],
    [128, 128, 0],
    [128, 0, 128],
    [0, 128, 128],
    [192, 192, 192],
    [128, 128, 128],
    [255, 165, 0],
    [128, 0, 128],
], dtype=np.uint8)

def read_npz(img_path, fold_ind, mode):
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    npz_path = os.path.join(r"D:\torchtestto\multimodel\17-coarse\semi\data\all-20\images_256_npz", f"fold_{fold_ind}", mode, base_name + ".npz")
    if not os.path.exists(npz_path):
        print(f"The file does not exist: {npz_path}")
        return None
    data = np.load(npz_path)
    return data['center_array'], data['adj'], data['edge_attr'], data['center_']

def create_graph_data(img_path, fold_ind, mode, purity=0.9, threshold=10, var_threshold=20):
    npz_data = read_npz(img_path, fold_ind, mode)
    if npz_data is not None:
        center_array, adj, edge_attr, center_ = npz_data
    else:
        img_data = cv2.imread(img_path)
        if img_data is None:
            raise ValueError(f"The image cannot be loaded: {img_path}")
        center_array, adj, edge_attr, center_ = granular_balls_generate(
            img_data, purity, threshold, var_threshold
        )

    x = torch.tensor(center_array, dtype=torch.float)
    edge_index = torch.tensor(adj, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    y = None
    num_classes = None

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_classes=num_classes)
    data.center_ = center_
    return data

def predict_image(model, data, device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1).cpu().numpy()
    return pred

def predictions_matrix(img, center_, pred_labels):
    h, w = img.shape[:2]
    point_matrix = np.zeros((h, w), dtype=np.uint8)
    region_matrix = np.zeros((h, w), dtype=np.uint8)

    for i, (x, y, Rx, Ry, *_) in enumerate(center_):
        x, y = int(x), int(y)
        Rx, Ry = int(Rx), int(Ry)
        cls = int(pred_labels[i])
        if 0 <= x < h and 0 <= y < w:
            point_matrix[x, y] = cls

        left = max(0, y - Rx)
        right = min(w - 1, y + Rx)
        up = max(0, x - Ry)
        down = min(h - 1, x + Ry)
        region_matrix[up:down + 1, left:right + 1] = cls

    return point_matrix, region_matrix

def visualize_predictions(img_path, point_matrix, region_matrix, save_path, base_name):
    output_img_P_L = os.path.join(save_path, base_name + "_P_L.png")
    output_img_P_RGB = os.path.join(save_path, base_name + "_P_RGB.png")
    output_img_L = os.path.join(save_path, base_name + "_L.png")
    output_img_RGB = os.path.join(save_path, base_name + "_RGB.png")

    Image.fromarray(point_matrix * 40, mode="L").save(output_img_P_L)
    Image.fromarray(region_matrix * 40, mode="L").save(output_img_L)

    new_path = save_path.replace("11_predictions", "11_mask_pred")
    os.makedirs(new_path, exist_ok=True)
    Image.fromarray(region_matrix, mode="L").save(os.path.join(new_path, base_name + ".png"))

    image = Image.open(img_path)
    image = np.array(image.convert("RGB"))

    color_mask = colors[point_matrix]
    vis_image = image.copy()
    mask = point_matrix != 0
    vis_image[mask] = color_mask[mask]
    Image.fromarray(vis_image, mode='RGB').save(output_img_P_RGB)

    color_mask = colors[region_matrix]
    vis_image = image.copy()
    mask = region_matrix != 0
    vis_image[mask] = color_mask[mask]
    Image.fromarray(vis_image, mode='RGB').save(output_img_RGB)

def main():
    parser = argparse.ArgumentParser(description='Spine Multiclass Classification')
    parser.add_argument('--image_dir', default=r"D:\torchtestto\multimodel\17-coarse\semi\data\all-20\images_256", type=str,
                        help='Input image directory')
    parser.add_argument('--split_dir', default=r"D:\torchtestto\multimodel\17-coarse\semi\data\all-20\h5py-5-200", type=str,
                        help='Directory of split index files')
    parser.add_argument('--purity', type=float, default=0.9, help='Granular-ball purity threshold')
    parser.add_argument('--threshold', type=float, default=10, help='Outlier threshold for granular-ball generation')
    parser.add_argument('--var_threshold', type=float, default=20, help='Variance threshold')
    parser.add_argument('--output_dir', type=str, default=r'D:\torchtestto\multimodel\17-coarse\semi\data\all-20\11_predictions',
                        help='Output directory')
    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for fold_ind in range(1, 6):
        print(f"\nFold {fold_ind}...")
        split_file = os.path.join(args.split_dir, f'split_ind_fold{fold_ind}.npz')
        if not os.path.exists(split_file):
            print(f"The index file does not exist: {split_file}")
            continue
        data = np.load(split_file)
        train_ids = data['train_ind'].tolist()
        val_ids = data['val_ind'].tolist()
        test_ids = data['test_ind'].tolist()

        num_classes = 17
        model = GCN_8_plus(num_features=25, num_classes=num_classes, initdim=16, inithead=16, edge_dim=3).to(device)
        model_path = f'checkpoint/best_model_957_17_fold{fold_ind}.pth'
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Successfully loaded model weights: {model_path}")
        except Exception as e:
            print(f"Failed to load model weights: {e}")
            continue

        for mode, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
            output_dir = os.path.join(args.output_dir, f"fold_{fold_ind}", mode)
            os.makedirs(output_dir, exist_ok=True)

            image_files = sorted([f for f in os.listdir(args.image_dir) if f.endswith(('.png', '.jpg'))])
            target_image_files = [f for f in image_files if int(f.split('_')[1]) in ids]
            if not target_image_files:
                print(f"Fold {fold_ind} {mode} set contains no image files")
                continue

            for img_file in tqdm(target_image_files, desc=f"Fold {fold_ind} {mode} prediction"):
                img_path = os.path.join(args.image_dir, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                try:
                    data = create_graph_data(img_path, fold_ind, mode, args.purity, args.threshold, args.var_threshold)
                except ValueError as e:
                    print(f"Failed to process image {img_file}: {e}")
                    continue

                pred_labels = predict_image(model, data, device)
                base_name = os.path.splitext(img_file)[0]
                output_path = os.path.join(output_dir, f"{base_name}_pred.npy")
                np.save(output_path, pred_labels)
                print(f"Predicted labels saved to: {output_path}")

                point_matrix, region_matrix = predictions_matrix(img, data.center_, pred_labels)
                visualize_predictions(img_path, point_matrix, region_matrix, output_dir, base_name)
                print(f"Fold {fold_ind} {mode} prediction visualization saved to: {output_dir}")


if __name__ == '__main__':
    main()
