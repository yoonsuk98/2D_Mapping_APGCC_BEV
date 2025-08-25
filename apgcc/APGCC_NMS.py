import os
import glob
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from config import cfg, merge_from_file
from models import build_model
from Wildtrack_dataset import Wildtrack
from Wildtrack_annotation import frameDataset  # homography 계산용

# ─── 경로 설정 파일 ─────────────────────────────────────
ROOT_WT       = './Wildtrack_dataset'
INPUT_FOLDER  = os.path.join(ROOT_WT, 'Image_subsets', 'C1')
SAVE_IMG_DIR  = './output/NMS/C1'
SAVE_BEV_DIR  = './output/NMS/BEV_frame'
SAVE_2D_BEV_DIR = './output/NMS/2D_frame'
POINT_LOG     = './output/NMS/C1/predicted_points.txt'
MODEL_WEIGHTS = './output/SHHA_best.pth'
CAM_INDEX     = 0
THRESHOLD     = 0.5

os.makedirs(SAVE_IMG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(POINT_LOG), exist_ok=True)
os.makedirs(SAVE_BEV_DIR, exist_ok=True)
os.makedirs(SAVE_2D_BEV_DIR, exist_ok=True)

# ─── 모델 로드 ─────────────────────────────────
cfg = merge_from_file(cfg, './configs/Wildtrack_test.yml')
cfg.test = True
cfg.OUTPUT_DIR = './output/'
model = build_model(cfg, training=False).cuda().eval()
state_dict = torch.load(MODEL_WEIGHTS, map_location='cpu')
model.load_state_dict(state_dict, strict=False)

# ─── Transforms 정의 ────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ─── Homography  정의 ──────────────────
wt = Wildtrack(ROOT_WT)
ds = frameDataset(wt, train=False,
                  world_reduce=4, img_reduce=12,
                  world_kernel_size=10, img_kernel_size=10)

scale = ds.img_reduce  # 12

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

def nms_on_bev_idx_scored(bev_idx_int, scores, radius_cells=4):
    
    if bev_idx_int is None or len(bev_idx_int) == 0:
        return np.array([], dtype=np.int32)

    bev_idx_int = np.asarray(bev_idx_int, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)
    assert len(bev_idx_int) == len(scores), "(bev_idx, scores) Do not match input length"

    # 점수 내림차순으로 우선순위 결정
    order = np.argsort(-scores)
    keep = []
    suppressed = np.zeros(len(bev_idx_int), dtype=bool)

    # 거리 계산
    bev_int = bev_idx_int.astype(np.float32)

    for k in order:
        if suppressed[k]:
            continue
        keep.append(k)
        d = np.linalg.norm(bev_int - bev_int[k], axis=1)
        suppressed |= (d < float(radius_cells))

    return np.array(keep, dtype=np.int32)

# ─── 프레임별 투영 & 시각화 ────────────────────
img_paths = sorted(glob.glob(os.path.join(INPUT_FOLDER, '*.*')))
# print(f"len(img_paths): {len(img_paths)}")
# print(f"len(ds): {len(ds)}")

with open(POINT_LOG, 'w') as log_fp:
    for idx in range(len(ds)):
        count = 0
        
        # GT 불러오기위한 설정 
        imgs, norm_world_pts, world_gt, imgs_gt, affine_mats, frame = ds[idx]
        img_path = os.path.join(INPUT_FOLDER, f"{frame:08d}.png")
        img = Image.open(img_path).convert('RGB')
        W, H = img.size

        if (H, W) != tuple(ds.img_shape):
            print("Do not match frameDataset & img_shape")

        img_tensor = transform(img)
        # print(f"[Frame {frame}] transform 이후 shape: {img_tensor.shape} (C,H,W)")

        # 1) APGCC 모델 추론
        inp = transform(img).unsqueeze(0).cuda()
        with torch.no_grad():
            out = model(inp)
        pts_norm_all = out['pred_points'][0].cpu().numpy()                 
        scores_all   = torch.softmax(out['pred_logits'][0], -1)[:,1].cpu().numpy() 

        mask    = scores_all > THRESHOLD
        pts_norm = pts_norm_all[mask]
        scores   = scores_all[mask]

        raw_cnt  = pts_norm_all.shape[0]
        kept_cnt = pts_norm.shape[0]

        # 2) (x,y) → (u,v) 픽셀 좌표 변환
        pix = []
        for x, y in pts_norm:
            u = int(x * W) if x <= 1.0 else int(x)
            v = int(y * H) if y <= 1.0 else int(y)
            pix.append((u, v))
        pix = np.array(pix, dtype=np.int32)
        # print(f"[Frame {frame}] Example of pixel : {pix[:5]}")
        # print(f"[Frame {frame}] raw_pts={raw_cnt}, kept_pts={kept_cnt}")

        # ─── APGCC 결과 시각화 ─────────────────────────
        H_map = H // scale
        W_map = W // scale
        ct_list = [ (u/scale, v/scale) for (u,v) in pix ]
        pred_heat_map = np.zeros((H_map, W_map), dtype=np.float32)
        kernel_size = ds.img_kernel_size
        for ct in ct_list:
            ct_int = np.array(ct, dtype=np.int32)
            if 0 <= ct_int[0] < W_map and 0 <= ct_int[1] < H_map:
                count += 1
            draw_umich_gaussian(pred_heat_map, ct_int, kernel_size/scale)

        fig = plt.figure()
        subplot0 = fig.add_subplot(211, title=f"Frame {frame} 2D imgs GT Heatmap")
        subplot1 = fig.add_subplot(212, title=f"Frame {frame} 2D imgs Pred Heatmap")
        subplot0.imshow(imgs_gt['heatmap'][CAM_INDEX].squeeze(0).numpy())
        subplot1.imshow(pred_heat_map)
        plt.savefig(os.path.join(SAVE_2D_BEV_DIR, f"frame_{frame:04d}_2D_heatmap.png"))
        plt.close(fig)

        # ─── 2차원 → 3차원 투영 ─────────────────────────
        H_mat = ds.world_from_img[CAM_INDEX].cpu().numpy()  # (3,3)
        pix_img_homo = np.concatenate([pix, np.ones((pix.shape[0], 1))], axis=1).T  # (3,N)
        
        # ─── 3차원 → BEV 그리드 좌표화 ───────────────────
        world_xy_homo = H_mat @ pix_img_homo
        world_xy = world_xy_homo[:2,:] / world_xy_homo[2:,:]
        x_grid = (world_xy[0] + 300) / 2.5 / ds.world_reduce
        y_grid = (world_xy[1] + 900) / 2.5 / ds.world_reduce

        bev_xy  = np.stack([x_grid, y_grid], axis=1).astype(np.float32)   
        bev_idx = np.rint(bev_xy).astype(np.int32)                      

        # ─── 점수 기반 BEV NMS  ──────────────────
        # 점수 순으로 우선 선택, 반경 내 나머지 억제
        r_cells = 3  
        if len(bev_idx) > 0:
            # 안전성: scores와 동일 길이/순서 체크
            assert len(bev_idx) == len(scores), "BEV 좌표와 scores 길이 불일치(마스크/순서 확인)"
            keep_idx = nms_on_bev_idx_scored(bev_idx, scores, radius_cells=r_cells)
            final_points = bev_idx[keep_idx].astype(np.int32)
            scores_kept  = scores[keep_idx]  # 필요시 활용
        else:
            final_points = np.empty((0, 2), dtype=np.int32)

        bev_idx = final_points 

        # ─── BEV 히트맵 생성/시각화 ───────────────────
        dx, dy = -243, -5
        H_bev, W_bev = ds.Rworld_shape
        bev_heatmap = np.zeros((H_bev, W_bev), dtype=np.uint8)

        if len(bev_idx) > 0:
            rows = bev_idx[:,1].astype(int) * 3
            cols = bev_idx[:,0].astype(int)
            rows += dx; cols += dy
            valid_mask = (rows >= 0) & (rows < H_bev) & (cols >= 0) & (cols < W_bev)
            rows = rows[valid_mask]; cols = cols[valid_mask]
            bev_heatmap[rows, cols] = 1

        key_point = norm_world_pts.squeeze().numpy()
        gt_map = np.zeros([120, 360])
        key_point[:, 0] = (key_point[:, 0] * 1440) / 4
        key_point[:, 1] = (key_point[:, 1] * 1440) / 4
        key_point[:, [0, 1]] = key_point[:, [1, 0]]
        key_point = key_point.astype(int)
        gt_map[key_point[:, 0], key_point[:, 1]] = 1
        
        fig2 = plt.figure()
        subplot2 = fig2.add_subplot(211, title=f"Frame {frame} BEV GT Heatmap")
        subplot3 = fig2.add_subplot(212, title=f"Frame {frame} BEV Pred Heatmap")
        subplot2.imshow(gt_map)
        subplot3.imshow(bev_heatmap)
        plt.title(f"Frame {frame} BEV Pred Heatmap")
        plt.savefig(os.path.join(SAVE_BEV_DIR, f"frame_{frame:04d}_BEV_heatmap.png"))
        plt.close(fig2)

        # ─────── 2D 원본 시각화 및 저장 ───────────────────
        vis = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for (u, v) in pix:
            cv2.circle(vis, (int(u), int(v)), radius=4, color=(0,255,0), thickness=-1)
        cv2.imwrite(os.path.join(SAVE_IMG_DIR, f"{frame:04d}.jpg"), vis)
        
        # log 저장
        log_fp.write(f"{frame:04d}.jpg {len(pix)} points\n")
        for u, v in pix:
            log_fp.write(f"{u} {v}\n")
        log_fp.write("\n")

print("All done APGCC_NMS")