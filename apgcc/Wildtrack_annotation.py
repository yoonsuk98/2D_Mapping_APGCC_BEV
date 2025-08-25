import os
import json
import time
from operator import itemgetter
import copy

import numpy as np
from PIL import Image
import kornia
from torchvision.datasets import VisionDataset
import torch
import torch.nn.functional as F
import torchvision.transforms as T
# from multiview_detector.utils.projection import *
import matplotlib.pyplot as plt

def random_affine(img, bboxs, pids, hflip=0.5, degrees=(-0, 0), translate=(.2, .2), scale=(0.6, 1.4), shear=(-0, 0),
                  borderValue=(128, 128, 128)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    border = 0  # width of added border (optional)
    height = img.shape[0]
    width = img.shape[1]

    # flipping
    F = np.eye(3)
    hflip = np.random.rand() < hflip
    if hflip:
        F[0, 0] = -1
        F[0, 2] = width

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(width / 2, height / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * width + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * height + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R @ F  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    n = bboxs.shape[0]
    area0 = (bboxs[:, 2] - bboxs[:, 0]) * (bboxs[:, 3] - bboxs[:, 1])

    # warp points
    xy = np.ones((n * 4, 3))
    xy[:, :2] = bboxs[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
    xy = (xy @ M.T)[:, :2].reshape(n, 8)

    # create new boxes
    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

    # apply angle-based reduction
    radians = a * math.pi / 180
    reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
    x = (xy[:, 2] + xy[:, 0]) / 2
    y = (xy[:, 3] + xy[:, 1]) / 2
    w = (xy[:, 2] - xy[:, 0]) * reduction
    h = (xy[:, 3] - xy[:, 1]) * reduction
    xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

    # reject warped points outside of image
    np.clip(xy[:, 0], 0, width - 1, out=xy[:, 0])
    np.clip(xy[:, 2], 0, width - 1, out=xy[:, 2])
    np.clip(xy[:, 1], 0, height - 1, out=xy[:, 1])
    np.clip(xy[:, 3], 0, height - 1, out=xy[:, 3])
    w = xy[:, 2] - xy[:, 0]
    h = xy[:, 3] - xy[:, 1]
    area = w * h
    ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
    i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

    bboxs = xy[i]
    pids = pids[i]

    return imw, bboxs, pids, M


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, sigma, k=1):
    radius = int(3 * sigma)
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=sigma)

    x, y = int(center[0]), int(center[1])

    H, W = heatmap.shape

    left, right = min(x, radius), min(W - x, radius + 1)
    top, bottom = min(y, radius), min(H - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


class img_color_denormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean).view([1, -1, 1, 1])
        self.std = torch.FloatTensor(std).view([1, -1, 1, 1])

    def __call__(self, tensor):
        return tensor * self.std.to(tensor.device) + self.mean.to(tensor.device)


def add_heatmap_to_image(heatmap, image):
    heatmap = cv2.resize(np.array(array2heatmap(heatmap)), (image.size))
    cam_result = np.uint8(heatmap * 0.3 + np.array(image) * 0.5)
    cam_result = Image.fromarray(cam_result)
    return cam_result


def array2heatmap(heatmap):
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-8)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, 2)
    heatmap = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    return heatmap

def project_2d_points(project_mat, input_points):
    vertical_flag = 0
    if input_points.shape[1] == 2:
        vertical_flag = 1
        input_points = np.transpose(input_points)
    input_points = np.concatenate([input_points, np.ones([1, input_points.shape[1]])], axis=0)
    output_points = project_mat @ input_points
    output_points = output_points[:2, :] / output_points[2, :]
    if vertical_flag:
        output_points = np.transpose(output_points)
    return output_points

def get_imgcoord_from_worldcoord_mat(intrinsic_mat, extrinsic_mat, z=0):
    """image of shape C,H,W (C,N_row,N_col); xy indexging; x,y (w,h) (n_col,n_row)
    world of shape N_row, N_col; indexed as specified in the dataset attribute (xy or ij)
    z in meters by default
    """
    threeD2twoD = np.array([[1, 0, 0], [0, 1, 0], [0, 0, z], [0, 0, 1]])
    project_mat = intrinsic_mat @ extrinsic_mat @ threeD2twoD
    return project_mat

def get_worldcoord_from_imgcoord_mat(intrinsic_mat, extrinsic_mat, z=0):
    """image of shape C,H,W (C,N_row,N_col); xy indexging; x,y (w,h) (n_col,n_row)
    world of shape N_row, N_col; indexed as specified in the dataset attribute (xy or ij)
    z in meters by default
    """
    project_mat = np.linalg.inv(get_imgcoord_from_worldcoord_mat(intrinsic_mat, extrinsic_mat, z))
    return project_mat

def get_worldcoord_from_imagecoord(image_coord, intrinsic_mat, extrinsic_mat, z=0):
    project_mat = get_worldcoord_from_imgcoord_mat(intrinsic_mat, extrinsic_mat, z)
    return project_2d_points(project_mat, image_coord)

def get_gt(Rshape, x_s, y_s, w_s=None, h_s=None, v_s=None, reduce=4, top_k=100, kernel_size=4):
    H, W = Rshape
    heatmap = np.zeros([1, H, W], dtype=np.float32)
    reg_mask = np.zeros([top_k], dtype=np.bool_)
    idx = np.zeros([top_k], dtype=np.int64)
    pid = np.zeros([top_k], dtype=np.int64)
    offset = np.zeros([top_k, 2], dtype=np.float32)
    wh = np.zeros([top_k, 2], dtype=np.float32)

    for k in range(len(v_s)):
        ct = np.array([x_s[k] / reduce, y_s[k] / reduce], dtype=np.float32)
        if 0 <= ct[0] < W and 0 <= ct[1] < H:
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(heatmap[0], ct_int, kernel_size / reduce)
            reg_mask[k] = 1
            idx[k] = ct_int[1] * W + ct_int[0]
            pid[k] = v_s[k]
            offset[k] = ct - ct_int
            if w_s is not None and h_s is not None:
                wh[k] = [w_s[k] / reduce, h_s[k] / reduce]
            # plt.imshow(heatmap[0])
            # plt.show()

    ret = {'heatmap': torch.from_numpy(heatmap), 'reg_mask': torch.from_numpy(reg_mask), 'idx': torch.from_numpy(idx),
           'pid': torch.from_numpy(pid), 'offset': torch.from_numpy(offset)}
    if w_s is not None and h_s is not None:
        ret.update({'wh': torch.from_numpy(wh)})
    return ret


import os
import json
import time
from operator import itemgetter
import copy

import numpy as np
from PIL import Image
import kornia
from torchvision.datasets import VisionDataset
import torch
import torch.nn.functional as F
import torchvision.transforms as T
# from multiview_detector.utils.projection import *
# from multiview_detector.utils.image_utils import draw_umich_gaussian, random_affine
import matplotlib.pyplot as plt


def get_gt(Rshape, x_s, y_s, w_s=None, h_s=None, v_s=None, reduce=4, top_k=100, kernel_size=4):
    H, W = Rshape
    heatmap = np.zeros([1, H, W], dtype=np.float32)
    reg_mask = np.zeros([top_k], dtype=np.bool_)
    idx = np.zeros([top_k], dtype=np.int64)
    pid = np.zeros([top_k], dtype=np.int64)
    offset = np.zeros([top_k, 2], dtype=np.float32)
    wh = np.zeros([top_k, 2], dtype=np.float32)

    for k in range(len(v_s)):
        ct = np.array([x_s[k] / reduce, y_s[k] / reduce], dtype=np.float32)
        if 0 <= ct[0] < W and 0 <= ct[1] < H:
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(heatmap[0], ct_int, kernel_size / reduce)
            reg_mask[k] = 1
            idx[k] = ct_int[1] * W + ct_int[0]
            pid[k] = v_s[k]
            offset[k] = ct - ct_int
            if w_s is not None and h_s is not None:
                wh[k] = [w_s[k] / reduce, h_s[k] / reduce]
            # plt.imshow(heatmap[0])
            # plt.show()

    ret = {'heatmap': torch.from_numpy(heatmap), 'reg_mask': torch.from_numpy(reg_mask), 'idx': torch.from_numpy(idx),
           'pid': torch.from_numpy(pid), 'offset': torch.from_numpy(offset)}
    if w_s is not None and h_s is not None:
        ret.update({'wh': torch.from_numpy(wh)})
    return ret


class frameDataset(VisionDataset):
    def __init__(self, base, train=True, reID=False, world_reduce=4, img_reduce=12,
                 world_kernel_size=10, img_kernel_size=10,
                 train_ratio=0.9, top_k=100, force_download=True,
                 semi_supervised=0.0, dropout=0.0, augmentation=False):
        super().__init__(base.root)

        self.base = base
        self.num_cam, self.num_frame = base.num_cam, base.num_frame
        # world (grid) reduce: on top of the 2.5cm grid
        self.reID, self.top_k = reID, top_k
        # reduce = input/output
        self.world_reduce, self.img_reduce = world_reduce, img_reduce
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col
        self.world_kernel_size, self.img_kernel_size = world_kernel_size, img_kernel_size
        self.semi_supervised = semi_supervised * train
        self.dropout = dropout
        self.transform = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    T.Resize((np.array(self.img_shape) * 8 // self.img_reduce).tolist())])
        self.augmentation = augmentation

        self.Rworld_shape = list(map(lambda x: x // self.world_reduce, self.worldgrid_shape))

        self.Rimg_shape = np.ceil(np.array(self.img_shape) / self.img_reduce).astype(int).tolist()

        if train:
            frame_range = range(0, 0)
        else:
            # frame_range = range(int(self.num_frame * train_ratio), self.num_frame)
            frame_range = range(0, self.num_frame)

        self.world_from_img, self.img_from_world = self.get_world_imgs_trans()
        self.world_from_img_17, self.img_from_world_17 = self.get_world_imgs_trans_17()
        world_masks = torch.ones([self.num_cam, 1] + self.worldgrid_shape)
        self.imgs_region = kornia.geometry.warp_perspective(world_masks, self.img_from_world, self.img_shape, 'nearest',
                                                   align_corners=False)

        self.img_fpaths = self.base.get_image_fpaths(frame_range)
        self.world_gt = {}
        self.imgs_gt = {}
        self.pid_dict = {}
        self.keeps = {}
        num_frame, num_world_bbox, num_imgs_bbox = 0, 0, 0
        num_keep, num_all = 0, 0
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                num_frame += 1
                keep = np.mean(np.array(frame_range) < frame) < self.semi_supervised if self.semi_supervised else 1
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                world_pts, world_pids = [], []
                img_bboxs, img_pids = [[] for _ in range(self.num_cam)], [[] for _ in range(self.num_cam)]
                if keep:
                    for pedestrian in all_pedestrians:
                        grid_x, grid_y = self.base.get_worldgrid_from_pos(pedestrian['positionID']).squeeze()
                        if pedestrian['personID'] not in self.pid_dict:
                            self.pid_dict[pedestrian['personID']] = len(self.pid_dict)
                        num_all += 1
                        num_keep += keep
                        num_world_bbox += keep
                        if self.base.indexing == 'xy':
                            world_pts.append((grid_x, grid_y))
                        else:
                            world_pts.append((grid_y, grid_x))
                        world_pids.append(self.pid_dict[pedestrian['personID']])
                        for cam in range(self.num_cam):
                            if itemgetter('xmin', 'ymin', 'xmax', 'ymax')(pedestrian['views'][cam]) != (-1, -1, -1, -1):
                                img_bboxs[cam].append(itemgetter('xmin', 'ymin', 'xmax', 'ymax')
                                                      (pedestrian['views'][cam]))
                                img_pids[cam].append(self.pid_dict[pedestrian['personID']])
                                num_imgs_bbox += 1
                self.world_gt[frame] = (np.array(world_pts), np.array(world_pids))
                self.imgs_gt[frame] = {}
                for cam in range(self.num_cam):
                    # x1y1x2y2
                    self.imgs_gt[frame][cam] = (np.array(img_bboxs[cam]), np.array(img_pids[cam]))
                self.keeps[frame] = keep

        print(f'all: pid: {len(self.pid_dict)}, frame: {num_frame}, keep ratio: {num_keep / num_all:.3f}\n'
              f'recorded: world bbox: {num_world_bbox / num_frame:.1f}, '
              f'imgs bbox per cam: {num_imgs_bbox / num_frame / self.num_cam:.1f}')
        # gt in mot format for evaluation
        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        if not os.path.exists(self.gt_fpath) or force_download:
            self.prepare_gt()
        pass

    def get_world_imgs_trans(self, z=0):
        # image and world feature maps from xy indexing, change them into world indexing / xy indexing (img)
        # world grid change to xy indexing
        Rworldgrid_from_worldcoord_mat = np.linalg.inv(self.base.worldcoord_from_worldgrid_mat @
                                                       self.base.world_indexing_from_xy_mat)

        # z in meters by default
        # projection matrices: img feat -> world feat
        worldcoord_from_imgcoord_mats = [get_worldcoord_from_imgcoord_mat(self.base.intrinsic_matrices[cam],
                                                                          self.base.extrinsic_matrices[cam],
                                                                          z / self.base.worldcoord_unit)
                                         for cam in range(self.num_cam)]
        # worldgrid(xy)_from_img(xy)
        proj_mats = [Rworldgrid_from_worldcoord_mat @ worldcoord_from_imgcoord_mats[cam] @ self.base.img_xy_from_xy_mat
                     for cam in range(self.num_cam)]
        world_from_img = torch.tensor(np.stack(proj_mats))
        # img(xy)_from_worldgrid(xy)
        img_from_world = torch.tensor(np.stack([np.linalg.inv(proj_mat) for proj_mat in proj_mats]))
        return world_from_img.float(), img_from_world.float()

    def get_world_imgs_trans_17(self, z=1.7):
            # image and world feature maps from xy indexing, change them into world indexing / xy indexing (img)
            # world grid change to xy indexing
            Rworldgrid_from_worldcoord_mat = np.linalg.inv(self.base.worldcoord_from_worldgrid_mat @
                                                        self.base.world_indexing_from_xy_mat)

            # z in meters by default
            # projection matrices: img feat -> world feat
            worldcoord_from_imgcoord_mats = [get_worldcoord_from_imgcoord_mat(self.base.intrinsic_matrices[cam],
                                                                            self.base.extrinsic_matrices[cam],
                                                                            z / self.base.worldcoord_unit)
                                            for cam in range(self.num_cam)]
            # worldgrid(xy)_from_img(xy)
            proj_mats = [Rworldgrid_from_worldcoord_mat @ worldcoord_from_imgcoord_mats[cam] @ self.base.img_xy_from_xy_mat
                        for cam in range(self.num_cam)]
            world_from_img = torch.tensor(np.stack(proj_mats))
            # img(xy)_from_worldgrid(xy)
            img_from_world = torch.tensor(np.stack([np.linalg.inv(proj_mat) for proj_mat in proj_mats]))
            return world_from_img.float(), img_from_world.float()

    def prepare_gt(self):
        og_gt = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)
            for single_pedestrian in all_pedestrians:
                def is_in_cam(cam):
                    return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                single_pedestrian['views'][cam]['xmax'] == -1 and
                                single_pedestrian['views'][cam]['ymin'] == -1 and
                                single_pedestrian['views'][cam]['ymax'] == -1)

                in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                if not in_cam_range:
                    continue
                grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID']).squeeze()
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def __getitem__(self, index, visualize=False):
        def plt_visualize():
            import cv2
            from matplotlib.patches import Circle
            fig, ax = plt.subplots(1)
            ax.imshow(img)
            for i in range(len(img_x_s)):
                x, y = img_x_s[i], img_y_s[i]
                if x > 0 and y > 0:
                    ax.add_patch(Circle((x, y), 10))
            plt.show()
            img0 = img.copy()
            for bbox in img_bboxs:
                bbox = tuple(int(pt) for pt in bbox)
                cv2.rectangle(img0, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            plt.imshow(img0)
            plt.show()

        frame = list(self.world_gt.keys())[index]

        #imgs
        imgs, imgs_gt, affine_mats, masks = [], [], [], []
        for cam in range(self.num_cam):
            t = self.img_fpaths[cam][frame]
            img = np.array(Image.open(self.img_fpaths[cam][frame]).convert('RGB'))
            img_bboxs, img_pids = self.imgs_gt[frame][cam]
            if self.augmentation:
                img, img_bboxs, img_pids, M = random_affine(img, img_bboxs, img_pids)
            else:
                M = np.eye(3)
            imgs.append(self.transform(img))
            affine_mats.append(torch.from_numpy(M).float())
            img_x_s, img_y_s = (img_bboxs[:, 0] + img_bboxs[:, 2]) / 2, img_bboxs[:, 3]
            img_w_s, img_h_s = (img_bboxs[:, 2] - img_bboxs[:, 0]), (img_bboxs[:, 3] - img_bboxs[:, 1])

            img_gt = get_gt(self.Rimg_shape, img_x_s, img_y_s, img_w_s, img_h_s, v_s=img_pids,
                            reduce=self.img_reduce, top_k=self.top_k, kernel_size=self.img_kernel_size)
            imgs_gt.append(img_gt)
            if visualize:
                plt_visualize()

        imgs = torch.stack(imgs)
        affine_mats = torch.stack(affine_mats)
        imgs_gt = {key: torch.stack([img_gt[key] for img_gt in imgs_gt]) for key in imgs_gt[0]}
        drop, keep_cams = np.random.rand() < self.dropout, torch.ones(self.num_cam, dtype=torch.bool)
        if drop:
            drop_cam = np.random.randint(0, self.num_cam)
            keep_cams[drop_cam] = 0
            for key in imgs_gt:
                imgs_gt[key][drop_cam] = 0
        # world gt
        world_pt_s, world_pid_s = self.world_gt[frame]
        world_gt = get_gt(self.Rworld_shape, world_pt_s[:, 0], world_pt_s[:, 1], v_s=world_pid_s,
                          reduce=self.world_reduce, top_k=self.top_k, kernel_size=self.world_kernel_size)

# x, y
        world_pt_s = world_pt_s.astype(np.float32)
        world_pt_s[:, 0] = world_pt_s[:, 0] / self.worldgrid_shape[1]
        world_pt_s[:, 1] = world_pt_s[:, 1] / self.worldgrid_shape[1]

        return imgs, torch.from_numpy(world_pt_s[None]), world_gt, imgs_gt, affine_mats, frame

    def __len__(self):

        return len(self.world_gt.keys())





