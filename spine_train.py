import argparse
from asyncore import write
from decimal import ConversionSyntax
import logging
from multiprocessing import reduction
import os
import random
import shutil
import sys
import time
import pdb
import cv2
import matplotlib.pyplot as plt
import imageio.v2 as imageio

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label

from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler)
from net_factory import net, net_factory
from utils import losses,ramps, val

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='D:\\torchtestto\multimodel\\17-coarse\semi\data\\all-20',
                    help='Name of Experiment')
parser.add_argument('--exp', type=str, default='GH', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=17, help='output channel of network')
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=14, help='labeled data')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float, default='6.0', help='magnitude')
parser.add_argument('--s_param', type=int, default=6, help='multinum of random masks')
parser.add_argument('--fold', type=int, default=5, help='fold number for cross-validation, 1-5')
parser.add_argument('--topk', type=int, default=4, help='Top-K uncertain patches for UMIX fusion')

args = parser.parse_args()

dice_loss = losses.DiceLoss(n_classes=17)

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))


def get_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_L(probs)
    return probs


def get_current_consistency_weight(epoch):
    return 5 * args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def get_L(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i]  # == c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)

        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()

def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)


def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x * 2 / 3), int(img_y * 2 / 3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w + patch_x, h:h + patch_y] = 0
    loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
    return mask.long(), loss_mask.long()


def random_mask(img, shrink_param=3):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    x_split, y_split = int(img_x / shrink_param), int(img_y / shrink_param)
    patch_x, patch_y = int(img_x * 2 / (3 * shrink_param)), int(img_y * 2 / (3 * shrink_param))
    mask = torch.ones(img_x, img_y).cuda()
    for x_s in range(shrink_param):
        for y_s in range(shrink_param):
            w = np.random.randint(x_s * x_split, (x_s + 1) * x_split - patch_x)
            h = np.random.randint(y_s * y_split, (y_s + 1) * y_split - patch_y)
            mask[w:w + patch_x, h:h + patch_y] = 0
            loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
    return mask.long(), loss_mask.long()


def contact_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_y = int(img_y * 4 / 9)
    h = np.random.randint(0, img_y - patch_y)
    mask[h:h + patch_y, :] = 0
    loss_mask[:, h:h + patch_y, :] = 0
    return mask.long(), loss_mask.long()


def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)  
    return loss_dice, loss_ce

def get_ema_alpha(iter_num, max_iter, alpha_min=0.99, alpha_max=0.999):
    """训练后期用更大的EMA decay，稳定伪标签"""
    return alpha_min + (alpha_max - alpha_min) * (iter_num / max_iter)
def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


import os
from collections import OrderedDict


def patients_to_slices(base_dir, fold, labelnum):
    list_file = os.path.join(base_dir, f"lists/train_fold{fold}.list")
    assert os.path.exists(list_file), f"{list_file} not found"

    with open(list_file, 'r') as f:
        slices = [line.strip() for line in f.readlines()]

    patient_dict = OrderedDict()
    for s in slices:
        name = os.path.basename(s)

        parts = name.split("_")
        assert parts[0] == "Patient", f"Unexpected slice name: {name}"
        pid = parts[1]

        patient_dict.setdefault(pid, []).append(s)

    patient_ids = list(patient_dict.keys())

    assert labelnum <= len(patient_ids), \
        f"labelnum={labelnum} > patients in this fold ({len(patient_ids)})"

    labeled_patients = patient_ids[:labelnum]

    labeled_slices = sum(len(patient_dict[pid]) for pid in labeled_patients)

    return labeled_slices

def get_coarse_mask_path(coarse_root, split, case):
    fname = case + ".png"
    path = os.path.join(coarse_root, split, fname)

    if not os.path.exists(path):
        raise FileNotFoundError(f"[CoarseMask Missing] {path}")

    return path

def umix_fusion(teacher_logits, student_logits, coarse_mask,
                   patch_size=16, alpha=0.5, conf_thresh=0.5,
                   t_high=0.7, t_low=0.3, num_classes=17):
    B, C, H, W = teacher_logits.shape
    device = teacher_logits.device

    with torch.no_grad():
        P_teacher = F.softmax(teacher_logits, dim=1)
        P_student = F.softmax(student_logits, dim=1)
        P_mean = 0.5 * (P_teacher + P_student)

        entropy = -torch.sum(P_mean * torch.log(P_mean + 1e-6), dim=1)
        entropy = entropy / np.log(C)

    ph, pw = H // patch_size, W // patch_size
    entropy_crop = entropy[:, :ph * patch_size, :pw * patch_size]
    U_patch = entropy_crop.view(B, ph, patch_size, pw, patch_size).mean(dim=(2, 4)).view(B, -1)

    coarse_onehot = F.one_hot(coarse_mask.clamp(0, C - 1), num_classes=C).permute(0, 3, 1, 2).float()
    C_patch = coarse_onehot[:, :, :ph * patch_size, :pw * patch_size]
    C_patch = C_patch.view(B, C, ph, patch_size, pw, patch_size).mean(dim=(3, 5))
    C_patch = torch.max(C_patch, dim=1)[0].view(B, -1)

    Conf_patch = alpha * C_patch + (1 - alpha) * (1 - U_patch)

    pseudo_mask = torch.zeros((B, H, W), device=device, dtype=torch.long)
    loss_mask = torch.zeros((B, H, W), device=device)

    teacher_conf, teacher_pred = torch.max(P_teacher, dim=1)

    for b in range(B):
        high_conf_patches = torch.where(Conf_patch[b] > conf_thresh)[0]

        for idx in high_conf_patches:
            idx = idx.item()
            i, j = idx // pw, idx % pw
            h0, h1 = i * patch_size, (i + 1) * patch_size
            w0, w1 = j * patch_size, (j + 1) * patch_size

            t_conf = teacher_conf[b, h0:h1, w0:w1]
            t_pred = teacher_pred[b, h0:h1, w0:w1]
            c_mask = coarse_mask[b, h0:h1, w0:w1]

            high_t = t_conf >= t_high
            pseudo_mask[b, h0:h1, w0:w1][high_t] = t_pred[high_t]
            loss_mask[b, h0:h1, w0:w1][high_t] = 1.0

            mid = (t_conf < t_high) & (t_conf >= t_low)
            if mid.any():
                c_onehot = F.one_hot(c_mask.clamp(0, C - 1), C).permute(2, 0, 1).float()
                P_fused = 0.5 * P_teacher[b, :, h0:h1, w0:w1] + 0.5 * c_onehot
                fused_pred = torch.argmax(P_fused, dim=0)
                pseudo_mask[b, h0:h1, w0:w1][mid] = fused_pred[mid]
                loss_mask[b, h0:h1, w0:w1][mid] = 0.5

        uncovered = loss_mask[b] == 0
        pseudo_mask[b][uncovered] = coarse_mask[b][uncovered]
        loss_mask[b][uncovered] = 0.3

    return pseudo_mask, loss_mask

def pre_train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path, '{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs / 2), int((args.batch_size - args.labeled_bs) / 2)

    model = net(in_chns=1, class_num=num_classes)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            fold=args.fold,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val", fold=args.fold)
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.fold, args.labelnum)

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            img_mask, loss_mask = generate_mask(img_a)
            gt_mixl = lab_a * img_mask + lab_b * (1 - img_mask)

            net_input = img_a * img_mask + img_b * (1 - img_mask)
            out_mixl = model(net_input)
            loss_dice, loss_ce = mix_loss(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)

            loss = (loss_dice + loss_ce) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f' % (iter_num, loss, loss_dice, loss_ce))

            if iter_num % 20 == 0:
                image = net_input[1, 0:1, :, :]
                writer.add_image('pre_train/Mixed_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_mixl, dim=1), dim=1, keepdim=True)
                writer.add_image('pre_train/Mixed_Prediction', outputs[1, ...] * 50, iter_num)
                labs = gt_mixl[1, ...].unsqueeze(0) * 50
                writer.add_image('pre_train/Mixed_GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_sum = np.zeros((num_classes - 1, 2))  # [dice, hd95]
                metric_count = np.zeros(num_classes - 1)
                for _, sampled_batch in enumerate(valloader):
                    image = sampled_batch["image"]
                    label = sampled_batch["label"]

                    print("val label shape:", label.shape)
                    print("val label min/max:", label.min().item(), label.max().item())
                    print("val label unique values:", torch.unique(label).cpu().numpy())
                    print("val label mean:", label.float().mean().item())
                    metric_i = val.test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model,
                        classes=num_classes
                    )
                    print("metric_i:", metric_i)
                    for cls_idx, (dice, hd95) in enumerate(metric_i):
                        if dice is None:
                            continue
                        metric_sum[cls_idx, 0] += dice
                        if hd95 is not None:
                            metric_sum[cls_idx, 1] += hd95
                        metric_count[cls_idx] += 1
                    metric_list = metric_sum / (metric_count[:, None] + 1e-8)

                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


def self_train(args, pre_snapshot_path, snapshot_path):
    coarse_root = os.path.join(
        args.root_path,
        "11_mask_pred",
        f"fold_{args.fold}"
    )
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path, '{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs / 2), int((args.batch_size - args.labeled_bs) / 2)

    model = net(in_chns=1, class_num=num_classes)
    ema_model = net(in_chns=1, class_num=num_classes, ema=True)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            fold=args.fold,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val", fold=args.fold)
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.fold, args.labelnum)

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    load_net(ema_model, pre_trained_model)
    load_net_opt(model, optimizer, pre_trained_model)
    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()
    ema_model.train()

    ce_loss = CrossEntropyLoss()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[
                                                                                               args.labeled_bs + unlabeled_sub_bs:]
            ulab_a, ulab_b = label_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], label_batch[
                                                                                              args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            case_batch = sampled_batch['case']
            u_case_a = case_batch[args.labeled_bs: args.labeled_bs + unlabeled_sub_bs]
            u_case_b = case_batch[args.labeled_bs + unlabeled_sub_bs:]

            with torch.no_grad():
                pre_a = ema_model(uimg_a)
                pre_b = ema_model(uimg_b)
                plab_a = get_masks(pre_a, nms=1)
                plab_b = get_masks(pre_b, nms=1)

                coarse_a = []
                coarse_b = []

                for case in u_case_a:
                    path = get_coarse_mask_path(coarse_root, "train", case)
                    cm = imageio.imread(path)
                    if cm.ndim == 3:
                        cm = cm[..., 0]
                    cm = cm.astype(np.int64)
                    assert cm.max() < args.num_classes, \
                        f"Invalid class id in coarse mask: {cm.max()}"
                    coarse_a.append(torch.from_numpy(cm).long())

                for case in u_case_b:
                    path = get_coarse_mask_path(coarse_root, "train", case)
                    cm = imageio.imread(path)
                    if cm.ndim == 3:
                        cm = cm[..., 0]
                    cm = cm.astype(np.int64)
                    assert cm.max() < args.num_classes, \
                        f"Invalid class id in coarse mask: {cm.max()}"
                    coarse_b.append(torch.from_numpy(cm).long())

                coarse_a = torch.stack(coarse_a).cuda()
                coarse_b = torch.stack(coarse_b).cuda()

                pseudo_a, umix_mask_a = umix_fusion(
                    teacher_logits=pre_a,
                    student_logits=model(uimg_a).detach(),
                    coarse_mask=coarse_a,
                )

                pseudo_b, umix_mask_b = umix_fusion(
                    teacher_logits=pre_b,
                    student_logits=model(uimg_b).detach(),
                    coarse_mask=coarse_b,
                )

                img_mask, cp_mask = generate_mask(img_a)
                loss_mask_a = cp_mask * umix_mask_a
                loss_mask_b = cp_mask * umix_mask_b

                pseudo_a = pseudo_a.clone()
                pseudo_a[pseudo_a < 0] = 0
                pseudo_b = pseudo_b.clone()
                pseudo_b[pseudo_b < 0] = 0

                unl_label = pseudo_a * img_mask + lab_a * (1 - img_mask)
                l_label = lab_b * img_mask + pseudo_b * (1 - img_mask)

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            net_input_unl = uimg_a * img_mask + img_a * (1 - img_mask)
            net_input_l = img_b * img_mask + uimg_b * (1 - img_mask)
            out_unl = model(net_input_unl)
            out_l = model(net_input_l)
            unl_dice, unl_ce = mix_loss(
                out_unl, pseudo_a, lab_a, loss_mask_a,
                u_weight=args.u_weight, unlab=True
            )

            l_dice, l_ce = mix_loss(
                out_l, lab_b, pseudo_b, loss_mask_b,
                u_weight=args.u_weight
            )

            loss_ce = unl_ce + l_ce
            loss_dice = unl_dice + l_dice

            loss = (loss_dice + loss_ce) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            alpha = get_ema_alpha(iter_num, max_iterations)
            update_model_ema(model, ema_model, alpha)

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f' % (iter_num, loss, loss_dice, loss_ce))

            if iter_num % 20 == 0:
                image = net_input_unl[1, 0:1, :, :]
                writer.add_image('train/Un_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_unl, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Un_Prediction', outputs[1, ...] * 50, iter_num)
                labs = unl_label[1, ...].unsqueeze(0) * 50
                writer.add_image('train/Un_GroundTruth', labs, iter_num)

                image_l = net_input_l[1, 0:1, :, :]
                writer.add_image('train/L_Image', image_l, iter_num)
                outputs_l = torch.argmax(torch.softmax(out_l, dim=1), dim=1, keepdim=True)
                writer.add_image('train/L_Prediction', outputs_l[1, ...] * 50, iter_num)
                labs_l = l_label[1, ...].unsqueeze(0) * 50
                writer.add_image('train/L_GroundTruth', labs_l, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_sum = np.zeros((num_classes - 1, 2))  # [dice, hd95]
                metric_count = np.zeros(num_classes - 1)

                for _, sampled_batch in enumerate(valloader):
                    metric_i = val.test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model,
                        classes=num_classes
                    )

                    for cls_idx, (dice, hd95) in enumerate(metric_i):
                        if dice is None:
                            continue
                        metric_sum[cls_idx, 0] += dice
                        if hd95 is not None:
                            metric_sum[cls_idx, 1] += hd95
                        metric_count[cls_idx] += 1

                metric_list = metric_sum / (metric_count[:, None] + 1e-8)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    pre_snapshot_path = "./model/{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_snapshot_path = "./model/{}_{}_labeled/self_train".format(args.exp, args.labelnum)
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    shutil.copy('D:\\torchtestto\multimodel\\17-coarse\semi\data\\all-20\\spine_train.py', self_snapshot_path)

    logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)