import os
import re
import glob
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import cfg, merge_from_file
from models import build_model
from Wildtrack_dataset import Wildtrack
from Wildtrack_annotation import frameDataset  # homography 계산용

# ─── 설정 ───────────────────────────────────────────
ROOT_WT       = './Wildtrack_dataset'
INPUT_PATH    = os.path.join(ROOT_WT, 'Image_subsets', 'C1')
SAVE_IMG_DIR  = './output/NMS_SORT/C1'
SAVE_BEV_DIR  = './output/NMS_SORT/BEV_frame'
POINT_LOG     = './output/NMS_SORT/predicted_points.txt'
MODEL_WEIGHTS = './output/SHHA_best.pth'
CAM_INDEX     = 0
THRESHOLD     = 0.5
FPS           = 10.0 

os.makedirs(SAVE_IMG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(POINT_LOG), exist_ok=True)
os.makedirs(SAVE_BEV_DIR, exist_ok=True)

# trail 길이 제한
MAX_TRAIL = 40  

# ─── NMS + SORT────────────────────────────────
def radius_nms_grid(points_xy, scores, radius=1.0):
    if len(points_xy) == 0:
        return np.empty((0,2), dtype=float), np.empty((0,), dtype=float)
    order = np.argsort(-scores)
    kept, used = [], np.zeros(len(points_xy), dtype=bool)
    pts = np.asarray(points_xy, dtype=float)
    for idx in order:
        if used[idx]:
            continue
        kept.append(idx)
        d = np.linalg.norm(pts - pts[idx], axis=1)
        used |= (d <= radius)
    kept = np.array(kept, dtype=int)
    return pts[kept], scores[kept]

class KalmanPoint:
    def __init__(self, x, y, track_id, dt=0.1, q=0.5, r=0.7):
        self.x = np.array([[x],[y],[0.0],[0.0]], dtype=float)
        self.P = np.eye(4) * 10.0
        self.id = track_id
        self.hits = 0
        self.no_updates = 0
        self.q = q; self.r = r
        self.set_mats(dt, q, r)

    def set_mats(self, dt, q, r):
        self.dt = dt; self.q = q; self.r = r
        self.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=float)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        dt2 = dt*dt; dt3 = dt2*dt; dt4 = dt2*dt2
        self.Q = q*np.array([[dt4/4,0,dt3/2,0],[0,dt4/4,0,dt3/2],[dt3/2,0,dt2,0],[0,dt3/2,0,dt2]], dtype=float)
        self.R = (r**2)*np.eye(2)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.no_updates += 1
        return self.x[:2].ravel()

    def innovation(self, z):
        z = np.asarray(z).reshape(2,1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return y, S

    def mahalanobis2(self, z):
        y, S = self.innovation(z)
        return float(y.T @ np.linalg.inv(S) @ y)

    def update(self, z):
        z = np.asarray(z).reshape(2,1)
        y, S = self.innovation(z)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        self.hits += 1
        self.no_updates = 0

class BevSortTracker:
    def __init__(self, max_age=20, min_hits=2, gate_chi2=5.99, dt=0.1, q=0.5, r=0.7):
        self.max_age = max_age
        self.min_hits = min_hits
        self.gate = gate_chi2
        self.tracks = []
        self.next_id = 1
        self.dt = dt; self.q = q; self.r = r

    def _associate_greedy(self, dets):
        if len(self.tracks)==0 or len(dets)==0:
            return [], list(range(len(self.tracks))), list(range(len(dets)))
        M, N = len(self.tracks), len(dets)
        cost = np.full((M,N), 1e6, dtype=float)
        for i,t in enumerate(self.tracks):
            for j,z in enumerate(dets):
                d2 = t.mahalanobis2(z)
                if d2 <= self.gate:
                    cost[i,j] = d2
        pairs, used_r, used_c = [], set(), set()
        flat = [(cost[i,j], i, j) for i in range(M) for j in range(N)]
        flat.sort(key=lambda x:x[0])
        for c,i,j in flat:
            if c >= 1e6: break
            if i in used_r or j in used_c: continue
            pairs.append((i,j)); used_r.add(i); used_c.add(j)
        unmatched_trk = [i for i in range(M) if i not in used_r]
        unmatched_det = [j for j in range(N) if j not in used_c]
        return pairs, unmatched_trk, unmatched_det

    def update(self, detections, dt=None):
        if dt is not None and dt > 0:
            self.dt = dt
            for t in self.tracks:
                t.set_mats(dt=self.dt, q=t.q, r=t.r)

        for t in self.tracks:
            t.predict()

        dets = np.asarray(detections, dtype=float)
        matches, un_trk, un_det = self._associate_greedy(dets)

        for i,j in matches:
            self.tracks[i].update(dets[j])

        for j in un_det:
            x,y = dets[j]
            self.tracks.append(KalmanPoint(x, y, self.next_id, dt=self.dt, q=0.5, r=0.7))
            self.next_id += 1

        # max_age 초과 트랙 제거
        self.tracks = [t for t in self.tracks if t.no_updates <= self.max_age]
        out = []
        for t in self.tracks:
            cx, cy = t.x[0,0], t.x[1,0]
            out.append((t.id, cx, cy, t.hits))
        return out

def id_to_color(tid: int):
    hue = (tid * 35) % 180
    hsv = np.uint8([[[hue, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

# ─── 전역 트래커/상태 ─────────────────────────────────
BEV_TRACKER = BevSortTracker(max_age=20, min_hits=2, gate_chi2=16.27, dt=1.0/FPS, q=1.0, r=3.0)
track_trails = {}  
_prev_frame_id = None

# ─── 모델/호모그래피 세팅 ───────────────────────────
cfg = merge_from_file(cfg, './configs/Wildtrack_test.yml')
cfg.test = True
cfg.OUTPUT_DIR = './output/'
model = build_model(cfg, training=False).cuda().eval()
state_dict = torch.load(MODEL_WEIGHTS, map_location='cpu')
model.load_state_dict(state_dict, strict=False)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

wt = Wildtrack(ROOT_WT)
ds = frameDataset(wt, train=False, world_reduce=4, img_reduce=12,
                  world_kernel_size=10, img_kernel_size=10)

H_mat = ds.world_from_img[CAM_INDEX].cpu().numpy()  
IMG_H, IMG_W = ds.img_shape  
WORLD_REDUCE = ds.world_reduce
Rworld_shape = ds.Rworld_shape  

dx, dy = -243, -5  # ROI
H_bev, W_bev = Rworld_shape

def to_frame_id(path):
    """파일명에서 숫자 프레임 추출. 없으면 None."""
    b = os.path.basename(path)
    m = re.search(r'(\d+)', b)
    return int(m.group(1)) if m else None

def process_image(img_path):
    global _prev_frame_id

    # GT 불러오기위한 설정 
    img = Image.open(img_path).convert('RGB')
    W_orig, H_orig = img.size
    if (H_orig, W_orig) != (IMG_H, IMG_W):
        img_for_model = img.resize((IMG_W, IMG_H), Image.BILINEAR)
    else:
        img_for_model = img

    # 1) APGCC 모델 추론
    inp = transform(img_for_model).unsqueeze(0).cuda()
    with torch.no_grad():
        out = model(inp)
    pts_norm_all = out['pred_points'][0].cpu().numpy()    
    scores_all   = torch.softmax(out['pred_logits'][0], -1)[:,1].cpu().numpy()

    mask = scores_all > THRESHOLD
    pts_norm = pts_norm_all[mask]
    scores   = scores_all[mask]

    # 2) 리사이즈 이미지 기준 픽셀 좌표
    W, H = IMG_W, IMG_H
    pix = []
    for x_n, y_n in pts_norm:
        u = int(x_n * W) if x_n <= 1.0 else int(x_n)
        v = int(y_n * H) if y_n <= 1.0 else int(y_n)
        pix.append((u, v))
    pix = np.array(pix, dtype=np.int32)

    # 3) 월드 투영
    if len(pix) > 0:
        pix_img_homo = np.concatenate([pix, np.ones((pix.shape[0], 1))], axis=1).T  # (3,N)
        world_xy_homo = H_mat @ pix_img_homo
        world_xy = world_xy_homo[:2,:] / np.clip(world_xy_homo[2:,:], 1e-6, None)  # (2,N)
    else:
        world_xy = np.zeros((2,0), dtype=float)

    # 4) BEV grid 연속 좌표
    x_grid_cont = (world_xy[0] + 300) / 2.5 / WORLD_REDUCE
    y_grid_cont = (world_xy[1] + 900) / 2.5 / WORLD_REDUCE
    bev_points_cont = np.stack([x_grid_cont, y_grid_cont], axis=1) if x_grid_cont.size>0 else np.empty((0,2))

    # 5) NMS → SORT 업데이트
    if bev_points_cont.shape[0] == 0:
        bev_pts_nms = np.empty((0,2), dtype=float)  
        
        # dt: 파일명에서 frame id 추정
        fid = to_frame_id(img_path)
        if fid is None:
            dt_est = 1.0 / FPS
        else:
            if _prev_frame_id is None:
                dt_est = 1.0 / FPS
            else:
                step = max(1, fid - _prev_frame_id)
                dt_est = step / FPS
            _prev_frame_id = fid
        tracks_bev = BEV_TRACKER.update([], dt=dt_est)
    else:
        bev_pts_nms, scores_nms = radius_nms_grid(bev_points_cont, scores, radius=2.0)
       
        # dt: 파일명에서 frame id 추정 
        fid = to_frame_id(img_path)
        if fid is None:
            dt_est = 1.0 / FPS
        else:
            if _prev_frame_id is None:
                dt_est = 1.0 / FPS
            else:
                step = max(1, fid - _prev_frame_id)
                dt_est = step / FPS
            _prev_frame_id = fid
        tracks_bev = BEV_TRACKER.update(bev_pts_nms, dt=dt_est)

    bev_heatmap_nms = np.zeros((H_bev, W_bev), dtype=np.uint8)
    if bev_pts_nms.shape[0] > 0:
        bev_idx_nms = np.rint(bev_pts_nms).astype(np.int32)
        rows_nms = bev_idx_nms[:, 1] * 3 + dx
        cols_nms = bev_idx_nms[:, 0] + dy
        vmask_nms = (rows_nms >= 0) & (rows_nms < H_bev) & (cols_nms >= 0) & (cols_nms < W_bev)
        bev_heatmap_nms[rows_nms[vmask_nms], cols_nms[vmask_nms]] = 1

    # 파일명/프레임 ID
    fid = to_frame_id(img_path)
    fid_str = f"{fid:04d}" if fid is not None else os.path.splitext(os.path.basename(img_path))[0]

    # 저장: NMS 통과 결과만 찍힌 BEV 히트맵
    cv2.imwrite(
        os.path.join(SAVE_BEV_DIR, f"frame_{fid_str}_BEV_heatmap.png"),
        (bev_heatmap_nms * 255).astype(np.uint8)
    )
    
    # ── 업데이트된 트랙만 활성 ───────────────────────────────────────────────────
    GRACE_AGE = 2  # 1~2 정도 권장
    active_tracks = [t for t in BEV_TRACKER.tracks if t.no_updates <= GRACE_AGE]
    active_ids = {t.id for t in active_tracks}

    # 관용 범위를 넘어선 트랙의 궤적만 제거
    for tid in list(track_trails.keys()):
        if tid not in active_ids:
            del track_trails[tid]

    # 궤적 포함 시각화
    bev_heatmap = np.zeros((H_bev, W_bev), dtype=np.uint8)
    if bev_points_cont.shape[0] > 0:
        bev_idx = np.round(bev_points_cont).astype(np.int32)
        rows = bev_idx[:,1] * 3 + dx
        cols = bev_idx[:,0] + dy
        vmask = (rows >= 0) & (rows < H_bev) & (cols >= 0) & (cols < W_bev)
        bev_heatmap[rows[vmask], cols[vmask]] = 1

    # 트랙 궤적 캔버스
    traj_canvas = np.zeros((H_bev, W_bev, 3), dtype=np.uint8)

    # ── active_track에서만 업데이트/그리기 ──────────────────────────
    if len(active_tracks) > 0:
        ids  = np.array([t.id     for t in active_tracks], dtype=int)
        xs_g = np.array([t.x[0,0] for t in active_tracks], dtype=float)
        ys_g = np.array([t.x[1,0] for t in active_tracks], dtype=float)

        rows_tr = (ys_g * 3).astype(int) + dx
        cols_tr = xs_g.astype(int) + dy
        vmask = (rows_tr >= 0) & (rows_tr < H_bev) & (cols_tr >= 0) & (cols_tr < W_bev)
        rows_tr = rows_tr[vmask]; cols_tr = cols_tr[vmask]; ids = ids[vmask]

        for r, c, tid in zip(rows_tr, cols_tr, ids):
            if tid not in track_trails:
                track_trails[tid] = []
            track_trails[tid].append((int(c), int(r)))  # (x=col, y=row)

            # trail 길이 제한
            if MAX_TRAIL is not None and len(track_trails[tid]) > MAX_TRAIL:
                track_trails[tid] = track_trails[tid][-MAX_TRAIL:]

    # 모든 궤적 그리기
    for tid, pts in track_trails.items():
        color = id_to_color(tid)
        if len(pts) >= 2:
            cv2.polylines(traj_canvas, [np.array(pts, dtype=np.int32)], isClosed=False, color=color, thickness=2)
            cv2.circle(traj_canvas, pts[-1], 3, color, -1)
        elif len(pts) == 1:
            cv2.circle(traj_canvas, pts[0], 3, color, -1)

    overlay = cv2.addWeighted(cv2.cvtColor((bev_heatmap*255).astype(np.uint8), cv2.COLOR_GRAY2BGR), 0.4,
                              traj_canvas, 1.0, 0)

    # 파일명/프레임 ID
    fid = to_frame_id(img_path)
    fid_str = f"{fid:04d}" if fid is not None else os.path.splitext(os.path.basename(img_path))[0]

    # 저장
    cv2.imwrite(os.path.join(SAVE_BEV_DIR, f"frame_{fid_str}_BEV_tracks.png"), traj_canvas)
    cv2.imwrite(os.path.join(SAVE_BEV_DIR, f"frame_{fid_str}_BEV_overlay.png"), overlay) # HEATMAP점 OVERLAY

    # 원 이미지에 점 표시
    vis = cv2.cvtColor(np.array(img_for_model), cv2.COLOR_RGB2BGR)
    for (u, v) in pix:
        cv2.circle(vis, (int(u), int(v)), 3, (0,255,0), -1)
    cv2.imwrite(os.path.join(SAVE_IMG_DIR, f"{fid_str}.jpg"), vis)

    return tracks_bev  


if __name__ == "__main__":
    if os.path.isdir(INPUT_PATH):
        exts = ('*.png','*.jpg','*.jpeg','*.bmp')
        img_list = []
        for e in exts:
            img_list.extend(glob.glob(os.path.join(INPUT_PATH, e)))
            
        img_list.sort(key=lambda p: (to_frame_id(p) is None, to_frame_id(p) if to_frame_id(p) is not None else p))
        print(f"[INFO] {len(img_list)} images found.")
        for p in img_list:
            tracks = process_image(p)
    else:
        print(f"[INFO] Single image: {INPUT_PATH}")
        tracks = process_image(INPUT_PATH)

    print("All done: BEV tracking visualizations saved.")