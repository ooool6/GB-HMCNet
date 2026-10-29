import argparse
import math
import os
import cv2
import numpy as np
from operator import add
from functools import reduce
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
import time
from collections import Counter
import networkx as nx
import scipy.sparse as sp
import itertools
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from torch_geometric.utils import degree

from img2graph import granular_balls_generate, cal_bound
from model import GCN_8_plus

random.seed(0)


def read_train_npz(img, fold_ind, mode="train"):
    base_name = os.path.splitext(img)[0]
    if mode == "train":
        npz_path = os.path.join(r"D:\torchtestto\multimodel\17-coarse\semi\data\all-20\images_256_npz", f"fold_{fold_ind}",
                                "train", base_name + ".npz")
    else:
        return None

    if not os.path.exists(npz_path):
        print(f"The file does not exist: {npz_path}")
        return None
    data = np.load(npz_path)
    center_array = data['center_array']
    adj = data['adj']
    edge_attr = data['edge_attr']
    center_ = data['center_']

    return center_array, adj, edge_attr, center_

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def assign_labels(center_, mask):
    labels = []
    for center in center_:
        x, y, Rx, Ry = center
        left, right, up, down = cal_bound(mask, [int(x), int(y)], Rx, Ry)
        region = mask[up:down + 1, left:right + 1]
        counts = Counter(region.flatten())
        max_count = -1
        label = -1
        for k, v in counts.items():
            if v > max_count and k != -1:
                max_count = v
                label = k
        labels.append(label)

    unique_labels = sorted(np.unique(labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = [label_map[l] for l in labels]
    return np.array(labels), len(unique_labels)


def filter_background_nodes(data):
    edge_index = data.edge_index.cpu().numpy()
    labels = data.y.cpu().numpy()

    num_nodes = data.num_nodes
    is_useful = np.zeros(num_nodes, dtype=bool)

    is_useful[labels != 0] = True

    adj_list = [[] for _ in range(num_nodes)]
    for src, dst in zip(*edge_index):
        adj_list[src].append(dst)
        adj_list[dst].append(src)

    for i in range(num_nodes):
        if labels[i] == 0:
            neighbors = adj_list[i]
            if any(labels[n] != 0 for n in neighbors):
                is_useful[i] = True

    old2new = {old: new for new, old in enumerate(np.where(is_useful)[0])}

    new_edge_index = []
    new_edge_attr = []
    for i, (src, dst) in enumerate(zip(*edge_index)):
        if is_useful[src] and is_useful[dst]:
            new_edge_index.append([old2new[src], old2new[dst]])
            new_edge_attr.append(data.edge_attr[i])

    data.x = data.x[is_useful]
    data.y = data.y[is_useful]
    data.edge_index = torch.tensor(new_edge_index, dtype=torch.long).t().contiguous()
    data.edge_attr = torch.stack(new_edge_attr)

    return data


def create_graph_data(img_file, mask, fold_ind, mode="train", purity=0.9, threshold=10, var_threshold=20):
    if mode == "train":
        result = read_train_npz(img_file, fold_ind, mode)
        if result is None:
            print(f"Skip the image {img_file},The npz file cannot be loaded.")
            return None
        center_array, adj, edge_attr, center_ = result
    else:
        img_path = os.path.join(r"D:\torchtestto\multimodel\17-coarse\semi\data\all-20\images_256", img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot read the image: {img_path}")
            return None
        center_array, adj, edge_attr, center_ = granular_balls_generate(img, purity, threshold, var_threshold)

    if mode in ["train", "val"]:
        labels, num_classes = assign_labels(center_, mask)
        y = torch.tensor(labels, dtype=torch.long)
    else:
        num_classes = 1
        y = None

    x = torch.tensor(center_array, dtype=torch.float)
    edge_index = torch.tensor(adj, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_classes=num_classes)
    return data

def load_data(fold_ind, train_ids, image_dir, mask_dir, purity, threshold, var_threshold, val_ratio=0.2):
    random.shuffle(train_ids)
    train_count = int(len(train_ids) * (1 - val_ratio))
    sub_train_ids = train_ids[:train_count]
    val_ids = train_ids[train_count:]

    train_data_list = []
    val_data_list = []
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])

    train_image_files = [f for f in image_files if int(f.split('_')[1]) in sub_train_ids]
    for img_file in train_image_files:
        mask_path = os.path.join(mask_dir, img_file)
        if mask_path.endswith('.jpg'):
            mask_path = mask_path.replace('.jpg', '.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"The mask cannot be loaded: {mask_path}")
            continue
        data = create_graph_data(img_file, mask, fold_ind, mode="train", purity=purity, threshold=threshold,
                                 var_threshold=var_threshold)
        if data is not None:
            train_data_list.append(data)

    val_image_files = [f for f in image_files if int(f.split('_')[1]) in val_ids]
    for img_file in val_image_files:
        mask_path = os.path.join(mask_dir, img_file)
        if mask_path.endswith('.jpg'):
            mask_path = mask_path.replace('.jpg', '.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"The mask cannot be loaded: {mask_path}")
            continue
        data = create_graph_data(img_file, mask, fold_ind, mode="val", purity=purity, threshold=threshold,
                                 var_threshold=var_threshold)
        if data is not None:
            val_data_list.append(data)

    return train_data_list, val_data_list, val_ids


def predict_and_evaluate(model, ids, image_dir, mask_dir, fold_ind, mode, purity, threshold, var_threshold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    data_list = []
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])
    target_image_files = [f for f in image_files if int(f.split('_')[1]) in ids]

    for img_file in target_image_files:
        mask = None
        if mode == "val":
            mask_path = os.path.join(mask_dir, img_file)
            if mask_path.endswith('.jpg'):
                mask_path = mask_path.replace('.jpg', '.png')
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"The mask cannot be loaded: {mask_path}")
                continue
        data = create_graph_data(img_file, mask, fold_ind, mode=mode, purity=purity, threshold=threshold,
                                 var_threshold=var_threshold)
        if data is not None:
            data_list.append((img_file, data))

    if not data_list:
        print(f"The {mode} set is empty, unable to perform prediction.")
        return None, None, None, None

    loader = DataLoader([data for _, data in data_list], batch_size=1, shuffle=False)

    if mode == "val":
        y_true, y_pred = [], []
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for data in tqdm(loader, desc=f"{mode} prediction", total=len(loader)):
                data = data.to(device)
                out = model(data)
                loss = criterion(out, data.y)
                total_loss += loss.item()
                pred = out.argmax(dim=1)
                y_true.extend(data.y.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())

        avg_loss = total_loss / len(loader)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        os.makedirs('results', exist_ok=True)
        result_path = f'results/predictions_fold{fold_ind}_val.txt'
        with open(result_path, 'w') as f:
            f.write(f'Fold {fold_ind} Validation Prediction Results:\n')
            f.write(f'Loss: {avg_loss:.4f}\n')
            f.write(f'Accuracy: {accuracy:.4f}\n')
            f.write(f'Precision: {precision:.4f}\n')
            f.write(f'Recall: {recall:.4f}\n')
            f.write(f'F1-score: {f1:.4f}\n')
        print(f'Fold {fold_ind} validation prediction results saved to {result_path}')

        return avg_loss, accuracy, precision, recall, f1
    else:
        predictions = []
        for img_file, data in tqdm(data_list, desc=f"{mode} prediction", total=len(data_list)):
            data = data.to(device)
            with torch.no_grad():
                out = model(data)
                pred = out.argmax(dim=1).cpu().numpy()
            predictions.append((img_file, pred.tolist()))

        os.makedirs('results', exist_ok=True)
        result_path = f'results/predictions_fold{fold_ind}_test_nodes.txt'
        with open(result_path, 'w') as f:
            f.write(f'Fold {fold_ind} Test Set Node Prediction Results:\n')
            for img_file, pred in predictions:
                f.write(f'Image: {img_file}, Node Predictions: {pred}\n')
        print(f'Fold {fold_ind} test set node prediction results saved to {result_path}')
        return None, None, None, None, None

def train_and_evaluate(train_data_list, val_data_list, fold_ind, hidden_dim=64, epochs=300, lr=0.01, deg_hist=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(train_data_list, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=1, shuffle=False)

    num_classes = max([data.num_classes for data in train_data_list])
    criterion = nn.CrossEntropyLoss()
    model = GCN_8_plus(num_features=25, num_classes=num_classes, initdim=16, inithead=16, edge_dim=3).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = 0
    patience = 30
    counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_y_true, train_y_pred = [], []
        for data in tqdm(train_loader, desc=f"train Epoch {epoch + 1}", total=len(train_loader)):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            pred = out.argmax(dim=1)
            train_y_true.extend(data.y.cpu().numpy())
            train_y_pred.extend(pred.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = accuracy_score(train_y_true, train_y_pred)
        train_precision = precision_score(train_y_true, train_y_pred, average='macro', zero_division=0)
        train_recall = recall_score(train_y_true, train_y_pred, average='macro', zero_division=0)
        train_f1 = f1_score(train_y_true, train_y_pred, average='macro', zero_division=0)

        model.eval()
        val_loss = 0
        val_y_true, val_y_pred = [], []
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"test Epoch {epoch + 1}", total=len(val_loader)):
                data = data.to(device)
                out = model(data)
                loss = criterion(out, data.y)
                val_loss += loss.item()
                pred = out.argmax(dim=1)
                val_y_true.extend(data.y.cpu().numpy())
                val_y_pred.extend(pred.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_y_true, val_y_pred)
        val_precision = precision_score(val_y_true, val_y_pred, average='macro', zero_division=0)
        val_recall = recall_score(val_y_true, val_y_pred, average='macro', zero_division=0)
        val_f1 = f1_score(val_y_true, val_y_pred, average='macro', zero_division=0)

        print(f'Fold {fold_ind} Epoch {epoch + 1}/{epochs}, '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.4f}, Train Precision: {train_precision:.4f}, '
              f'Train Recall: {train_recall:.4f}, Train F1-score: {train_f1:.4f}, '
              f'Validation Loss: {avg_val_loss:.4f}, '
              f'Validation Accuracy: {val_accuracy:.4f}, Validation Precision: {val_precision:.4f}, '
              f'Validation Recall: {val_recall:.4f}, Validation F1-score: {val_f1:.4f}')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_path = f'checkpoint/best_model_957_17_fold{fold_ind}.pth'
            os.makedirs('checkpoint', exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(
                f'Fold {fold_ind} Epoch {epoch + 1}: Best model saved to {best_model_path}, Validation F1-score: {best_val_f1:.4f}')
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            print(f'Fold {fold_ind} Epoch {epoch + 1}: No improvement in validation F1-score, early stopping.')
            break
    return model, best_model_path


def main():
    parser = argparse.ArgumentParser(description='Spine CT Multiclass Segmentation')
    parser.add_argument('--image_dir', default=r"D:\torchtestto\multimodel\17-coarse\semi\data\all-20\images_256", type=str,
                        help='Directory of CT images')
    parser.add_argument('--mask_dir', default=r"D:\torchtestto\multimodel\17-coarse\semi\data\all-20\masks_11_256",
                        type=str, help='Directory of mask labels')
    parser.add_argument('--split_dir', default=r"D:\torchtestto\multimodel\17-coarse\semi\data\all-20\h5py-5-200", type=str,
                        help='Directory of split index files')
    parser.add_argument('--purity', type=float, default=0.9, help='Granular-ball purity threshold')
    parser.add_argument('--threshold', type=float, default=10, help='Outlier threshold for granular-ball generation')
    parser.add_argument('--var_threshold', type=float, default=20, help='Variance threshold')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio within the training set')
    args = parser.parse_args()

    for fold_ind in range(1, 6):
        print(f"\nProcessing fold {fold_ind}...")
        split_file = os.path.join(args.split_dir, f'split_ind_fold{fold_ind}.npz')
        if not os.path.exists(split_file):
            print(f"Split index file does not exist: {split_file}")
            continue
        data = np.load(split_file)
        train_ids = data['train_ind'].tolist()
        test_ids = data['test_ind'].tolist()

        train_data_list, val_data_list, val_ids = load_data(fold_ind, train_ids, args.image_dir, args.mask_dir,
                                                            args.purity, args.threshold, args.var_threshold,
                                                            args.val_ratio)
        if not train_data_list or not val_data_list:
            print(f"Fold {fold_ind}: No valid training or validation data loaded!")
            continue

        model, best_model_path = train_and_evaluate(train_data_list, val_data_list, fold_ind,
                                                    hidden_dim=args.hidden_dim, epochs=args.epochs, lr=args.lr,
                                                    deg_hist=None)

        num_classes = max([data.num_classes for data in train_data_list])
        model = GCN_8_plus(num_features=25, num_classes=num_classes, initdim=16, inithead=16, edge_dim=3).to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(torch.load(best_model_path))

        val_metrics = predict_and_evaluate(model, val_ids, args.image_dir, args.mask_dir, fold_ind, "val", args.purity,
                                           args.threshold, args.var_threshold)
        if val_metrics[0] is not None:
            print(
                f'Fold {fold_ind} Validation Set: Loss: {val_metrics[0]:.4f}, '
                f'Accuracy: {val_metrics[1]:.4f}, Precision: {val_metrics[2]:.4f}, '
                f'Recall: {val_metrics[3]:.4f}, F1-score: {val_metrics[4]:.4f}')
        test_metrics = predict_and_evaluate(model, test_ids, args.image_dir, args.mask_dir, fold_ind, "test",
                                            args.purity, args.threshold, args.var_threshold)


def compute_degree_histogram(data_list, max_deg=100):
    deg_hist = torch.zeros(max_deg, dtype=torch.long)
    for data in data_list:
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
        deg = deg.clamp(max=max_deg - 1)
        deg_hist += torch.bincount(deg, minlength=max_deg)
    return deg_hist


if __name__ == '__main__':
    main()