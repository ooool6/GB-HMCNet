import argparse
import os
import shutil

import cv2
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='D:\\torchtestto\multimodel\\17-coarse\semi\data\\all-20', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='GH', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=17, help='output channel of network')
parser.add_argument('--labelnum', type=int, default=14, help='labeled data')
parser.add_argument('--stage_name', type=str, default='self_train', help='self or pre')
parser.add_argument('--fold', type=int, default=5)


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, jc, hd95, asd


def test_single_volume(case, net, test_save_path, FLAGS):
    h5_path = os.path.join(FLAGS.root_path, "data", "slices", f"{case}.h5")
    with h5py.File(h5_path, 'r') as h5f:
        image = h5f['image'][:]   # (H, W)
        label = h5f['label'][:]   # (H, W)

    x, y = image.shape
    image_rs = zoom(image, (256 / x, 256 / y), order=0)

    input = torch.from_numpy(image_rs).unsqueeze(0).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        out = net(input)
        if isinstance(out, (list, tuple)):
            out = out[0]
        out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
        out = out.cpu().numpy()

    pred = zoom(out, (x / 256, y / 256), order=0)

    save_path = os.path.join(test_save_path, f"{case}.png")
    cv2.imwrite(save_path, pred.astype(np.uint8))

    metric_list = []
    for cls in range(1, FLAGS.num_classes):
        if np.sum(label == cls) == 0:
            continue
        if np.sum(pred == cls) == 0:
            metric_list.append((0, 0, 0, 0))
        else:
            metric_list.append(
                calculate_metric_percase(pred == cls, label == cls)
            )

    return metric_list

def Inference(FLAGS):
    test_list = os.path.join(
        FLAGS.root_path,
        f"lists/test_fold{FLAGS.fold}.list"
    )
    assert os.path.exists(test_list)

    with open(test_list, 'r') as f:
        image_list = [line.strip() for line in f.readlines()]

    image_list = sorted(image_list)

    snapshot_path = os.path.join(
        FLAGS.root_path,
        "model",
        "GH",
        f"{FLAGS.exp}_{FLAGS.labelnum}_labeled",
        f"fold{FLAGS.fold}",
        FLAGS.stage_name
    )

    test_save_path = "./model/GH/{}_{}_labeled/{}_predictions/".format(FLAGS.exp, FLAGS.labelnum, FLAGS.fold,FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
    save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_model_path))

    print("init weight from {}".format(save_model_path))
    net.eval()

    all_metrics = []
    for case in tqdm(image_list):
        case_metrics = test_single_volume(case, net, test_save_path, FLAGS)
        all_metrics.extend(case_metrics)

    avg_metric = np.mean(np.asarray(all_metrics), axis=0)

    return avg_metric, test_save_path


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric, test_save_path = Inference(FLAGS)
    print(metric)
    print("Average Dice:", metric[0])
    with open(os.path.join(test_save_path, '../performance.txt'), 'w') as f:
        f.write(f"Mean Dice: {metric[0]:.4f}\n")
        f.write(f"Mean Jaccard: {metric[1]:.4f}\n")
        f.write(f"Mean HD95: {metric[2]:.4f}\n")
        f.write(f"Mean ASD: {metric[3]:.4f}\n")
