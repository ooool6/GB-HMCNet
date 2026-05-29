def calculate_metric_percase(pred, gt):
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)

    if gt.sum() == 0:
        return None, None   

    dice = metric.binary.dc(pred, gt)

    if pred.sum() == 0:
        hd95 = None
    else:
        hd95 = metric.binary.hd95(pred, gt)

    return dice, hd95


def test_single_volume(image, label, model, classes, patch_size=[256, 256]):

    if image.dim() == 3:          # [1, H, W]
        image = image.unsqueeze(1)  # -> [1, 1, H, W]
    if label.dim() == 3:
        label = label.squeeze(0)   # -> [H, W]

    image = image.cuda()
    label = label.cpu().numpy()

    model.eval()
    with torch.no_grad():
        output = model(image)
        if isinstance(output, (list, tuple)):
            output = output[0]

        pred = torch.argmax(torch.softmax(output, dim=1), dim=1)
        pred = pred.squeeze(0).cpu().numpy()  # [H, W]

    metric_list = []
    for i in range(1, classes):
        dice, hd95 = calculate_metric_percase(pred == i, label == i)
        if dice is None:
            continue
        metric_list.append((dice, hd95))

    return metric_list



def test_single_volume_cross(image, label, model_l, model_r, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        model_r.eval()
        model_l.eval()
        with torch.no_grad():
            output_l = model_l(input)
            output_r = model_r(input)
            output = (output_l + output_r) / 2
            if len(output)>1:
                output = output[0]
            out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list
