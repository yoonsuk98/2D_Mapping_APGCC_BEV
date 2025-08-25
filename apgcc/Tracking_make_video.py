import cv2
import glob
import re
import os

def make_video_from_frames(img_dir, output_path, fps=10):
    pattern = os.path.join(img_dir, 'frame_*_BEV_overlay.png')
    files = glob.glob(pattern)
    files.sort(key=lambda p: int(re.search(r'frame_(\d+)_BEV_overlay', os.path.basename(p)).group(1)))
    
    if not files:
        raise RuntimeError(f"No images found in {img_dir}")

    frame0 = cv2.imread(files[0])
    if frame0 is None:
        raise RuntimeError(f"Failed to read image {files[0]}")
    H, W = frame0.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for path in files:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: failed to load {path}, skipping")
            continue
        writer.write(img)

    writer.release()
    print(f"Saved video: {output_path}")

if __name__ == "__main__":
    method = "NMS_SORT" # only "NMS_SORT"
    img_directory = './output/'+method+'/BEV_frame'
    output_video = 'Tracking_result.mp4'
    make_video_from_frames(img_directory, output_video, fps=10)