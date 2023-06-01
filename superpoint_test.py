from pathlib import Path
import torch
from torch import nn
# from mmcv.ops.point_sample import bilinear_grid_sample
import torch.nn.functional as F



def bilinear_grid_sample(im, grid, align_corners=False):
    """Given an input and a flow-field grid, computes the output using input
    values and pixel locations from grid. Supported only bilinear interpolation
    method to sample the input pixels.

    Args:
        im (torch.Tensor): Input feature map, shape (N, C, H, W)
        grid (torch.Tensor): Point coordinates, shape (N, Hg, Wg, 2)
        align_corners {bool}: If set to True, the extrema (-1 and 1) are
            considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.
    Returns:
        torch.Tensor: A tensor with sampled points, shape (N, C, Hg, Wg)
    """
    n, c, h, w = im.shape
    gn, gh, gw, _ = grid.shape
    assert n == gn

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]

    if align_corners:
        x = ((x + 1) / 2) * (w - 1)
        y = ((y + 1) / 2) * (h - 1)
    else:
        x = ((x + 1) * w - 1) / 2
        y = ((y + 1) * h - 1) / 2

    x = x.view(n, -1)
    y = y.view(n, -1)

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
    wb = ((x1 - x) * (y - y0)).unsqueeze(1)
    wc = ((x - x0) * (y1 - y)).unsqueeze(1)
    wd = ((x - x0) * (y - y0)).unsqueeze(1)

    # Apply default for grid_sample function zero padding
    im_padded = F.pad(im, pad=[1, 1, 1, 1], mode='constant', value=0)
    padded_h = h + 2
    padded_w = w + 2
    # save points positions after padding
    x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

    # Clip coordinates to padded image size
    x0 = torch.where(x0 < 0, torch.tensor(0), x0)
    x0 = torch.where(x0 > padded_w - 1, torch.tensor(padded_w - 1), x0)
    x1 = torch.where(x1 < 0, torch.tensor(0), x1)
    x1 = torch.where(x1 > padded_w - 1, torch.tensor(padded_w - 1), x1)
    y0 = torch.where(y0 < 0, torch.tensor(0), y0)
    y0 = torch.where(y0 > padded_h - 1, torch.tensor(padded_h - 1), y0)
    y1 = torch.where(y1 < 0, torch.tensor(0), y1)
    y1 = torch.where(y1 > padded_h - 1, torch.tensor(padded_h - 1), y1)

    im_padded = im_padded.view(n, c, -1)

    x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)

    Ia = torch.gather(im_padded, 2, x0_y0)
    Ib = torch.gather(im_padded, 2, x0_y1)
    Ic = torch.gather(im_padded, 2, x1_y0)
    Id = torch.gather(im_padded, 2, x1_y1)

    return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)


def max_pool(x, nms_radius: int):
    return torch.nn.functional.max_pool2d(
        x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    zeros = torch.zeros_like(scores)
    #max_mask = scores == max_pool(scores, nms_radius)
    max_mask = torch.abs(scores - max_pool(scores, nms_radius)) < 1e-6
    for _ in range(2):
        supp_mask = max_pool(max_mask.float(), nms_radius) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        #new_max_mask = supp_scores == max_pool(supp_scores, nms_radius)
        new_max_mask = torch.abs(supp_scores - max_pool(supp_scores, nms_radius)) < 1e-6
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)]).to(keypoints).unsqueeze(0)
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    descriptors = bilinear_grid_sample(descriptors, keypoints.view(b, 1, -1, 2), align_corners=True)
    # descriptors = F.grid_sample(input=descriptors, grid=keypoints.view(b, 1, -1, 2), mode='bilinear', padding_mode='border', align_corners=True)
    # descriptors = torch.nn.functional.grid_sample(
    #     descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', align_corners=True)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2., dim=1)
    return descriptors


class SuperPoint(torch.nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.descriptor_dim = self.config['descriptor_dim']
        self.nms_radius = self.config['nms_radius']
        self.keypoint_threshold = self.config['keypoint_threshold']
        self.max_keypoints = self.config['max_keypoints']
        self.remove_borders = self.config['remove_borders']

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.descriptor_dim,
            kernel_size=1, stride=1, padding=0)


        mk = self.max_keypoints
        if mk == 0 or mk < -1:
            raise ValueError('"max_keypoints" must be positive or "-1"')

        print('Loaded SuperPoint model')


    def forward(self, image):
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        #scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        #scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)

        scores = torch.nn.functional.pixel_shuffle(scores, 8)
        scores = simple_nms(scores, self.nms_radius)
        scores = scores.squeeze(1)

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2., dim=1)

        keypoints = []
        scores_out = []
        descriptors_out = []
        for i in range(b):
            # Extract keypoints
            s = scores[i]
            k = torch.nonzero(s > self.keypoint_threshold)
            s = s[s > self.keypoint_threshold]

            # Discard keypoints near the image borders
            k, s = remove_borders(k, s, self.remove_borders, h * 8, w * 8)

            # Keep the k keypoints with highest score
            if self.max_keypoints >= 0:
                k, s = top_k_keypoints(k, s, self.max_keypoints)

            # Convert (h, w) to (x, y)
            k = torch.flip(k, [1]).float()

            # Extract descriptors
            descriptors_out.append(sample_descriptors(k.unsqueeze(0), descriptors[i].unsqueeze(0), 8)[0])
            keypoints.append(k)
            scores_out.append(s)

        return {
            'keypoints': keypoints,
            'scores': scores_out,
            'descriptors': descriptors_out,
        }
