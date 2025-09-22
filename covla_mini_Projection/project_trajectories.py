#!/usr/bin/env python3
"""
Project CoVLA 3D trajectories onto images and export annotated frames and videos.
Shows current trajectory and next 20 future trajectories for trajectory prediction visualization.

Assumptions
- Dataset root has subfolders: images/<sequence>/<frame>.png, states/<sequence>.jsonl, video_samples/<sequence>.mp4 (optional)
- Each JSONL line has keys: intrinsic_matrix (3x3), extrinsic_matrix (4x4), image_path (relative to dataset root), trajectory (Nx3), frame_id
- image_size and frequency are supplied via CLI or a small JSON (default: width=1928, height=1208, fps=20)


Usage (example)
  python project_trajectories.py --dataset-root "c:/Users/USER/Pictures/BASEPIC/CoVLA-Dataset-Mini" --fps 20 --width 1928 --height 1208 --num-future-trajectories 20

Outputs
- Annotated images in <dataset-root>/overlays/<sequence>/<frame>.png showing current + future trajectories
- Video in <dataset-root>/overlays/<sequence>.mp4

Visualization Features
- Current trajectory: Bright green with red start dot
- Future trajectories: Color gradient from green (near future) to red (far future)
- Trajectory thickness decreases for far future trajectories
- Legend and frame information displayed on each image
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# ----------------------- Math helpers -----------------------

def to_homogeneous(points_xyz: np.ndarray) -> np.ndarray:
    """Convert (N,3) to homogeneous (N,4)."""
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must be (N,3)")
    ones = np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)
    return np.concatenate([points_xyz, ones], axis=1)


def project_points(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project camera-frame 3D points (N,3) to image pixels (N,2) using intrinsics K.
    Points with Z<=0 will be filtered out by caller.
    """
    x = points_cam[:, 0] / points_cam[:, 2]
    y = points_cam[:, 1] / points_cam[:, 2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * x + cx
    v = fy * y + cy
    return np.stack([u, v], axis=1)


def world_to_camera(points_world: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    """Transform world (vehicle) frame points to camera frame using extrinsic matrix.

    After analysis, the CoVLA extrinsic matrix appears to be the transformation 
    from vehicle frame to camera frame (not camera to vehicle as initially thought).
    
    The camera appears to be mounted with a downward tilt, and trajectory points
    at ground level project above the image center, which suggests we should use
    the extrinsic matrix directly rather than its inverse.
    """
    if extrinsic.shape == (4, 4):
        T_veh_to_cam = extrinsic
    else:
        raise ValueError("extrinsic_matrix must be 4x4")

    # Use the extrinsic matrix directly for vehicle -> camera transformation
    # Convert points to homogeneous coordinates and transform
    pts_h = to_homogeneous(points_world.astype(np.float64))
    pts_cam_h = (T_veh_to_cam @ pts_h.T).T
    return pts_cam_h[:, :3]


# ----------------------- Drawing helpers -----------------------

def draw_trajectory(img: np.ndarray, uv: np.ndarray, color=(0, 255, 0), thickness=2) -> np.ndarray:
    """Draw a polyline trajectory on image. uv is (N,2) float pixel coordinates."""
    h, w = img.shape[:2]
    # Keep only points that lie within the image for drawing connectivity purposes
    uv_int = np.round(uv).astype(int)
    # Draw as segments between consecutive valid points
    prev = None
    for p in uv_int:
        x, y = int(p[0]), int(p[1])
        if 0 <= x < w and 0 <= y < h:
            if prev is not None:
                cv2.line(img, prev, (x, y), color, thickness, cv2.LINE_AA)
            prev = (x, y)
        else:
            prev = None
    return img


def draw_trajectory_waypoints(img: np.ndarray, uv: np.ndarray, color=(0, 0, 255), radius=4) -> np.ndarray:
    """Draw trajectory waypoints as thick circular dots on image. uv is (N,2) float pixel coordinates."""
    h, w = img.shape[:2]
    uv_int = np.round(uv).astype(int)
    
    for p in uv_int:
        x, y = int(p[0]), int(p[1])
        if 0 <= x < w and 0 <= y < h:
            # Draw thick circular waypoint
            cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)
            # Add white border for better visibility
            cv2.circle(img, (x, y), radius + 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    return img


def get_future_trajectory_color(future_index: int, max_futures: int) -> Tuple[int, int, int]:
    """Get color for future trajectory based on how far in the future it is.
    Returns BGR color tuple.
    """
    # Color gradient from green (current) to red (far future)
    ratio = future_index / max(max_futures - 1, 1)
    
    # HSV color space: Hue from 60 (green) to 0 (red)
    hue = int(60 * (1 - ratio))  # 60 = green, 0 = red
    saturation = 255
    value = 255
    
    # Convert HSV to BGR
    hsv = np.uint8([[[hue, saturation, value]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(map(int, bgr))


def draw_multiple_trajectories(img: np.ndarray, trajectories_uv: List[np.ndarray], 
                             current_traj_uv: np.ndarray = None) -> np.ndarray:
    """Draw multiple future trajectories with thick red waypoints for accurate visual evaluation."""
    
    # Draw future trajectories as thick red waypoints (as requested by user)
    for i, traj_uv in enumerate(trajectories_uv):
        if traj_uv.size > 0:
            # All future trajectory waypoints in thick red for quality visual evaluation
            radius = max(3, 5 - i // 5)  # Slightly decrease size for far future trajectories
            draw_trajectory_waypoints(img, traj_uv, color=(0, 0, 255), radius=radius)
    
    # Draw current trajectory as connected green line with waypoints
    if current_traj_uv is not None and current_traj_uv.size > 0:
        # Draw connecting line in green
        draw_trajectory(img, current_traj_uv, color=(0, 255, 0), thickness=2)
        # Draw current trajectory waypoints as green dots
        draw_trajectory_waypoints(img, current_traj_uv, color=(0, 255, 0), radius=3)
        # Add a larger blue dot at the start of current trajectory (vehicle position)
        if len(current_traj_uv) > 0:
            p0 = tuple(np.round(current_traj_uv[0]).astype(int))
            cv2.circle(img, p0, 8, (255, 0, 0), -1, cv2.LINE_AA)  # Blue dot for vehicle
    
    return img


def put_info(img: np.ndarray, text: str, org=(20, 40)) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)


# ----------------------- Core logic -----------------------

def process_sequence(dataset_root: Path, jsonl_path: Path, out_root: Path, width: int, height: int, fps: int, num_future_trajectories: int = 20) -> None:
    """Process one states/*.jsonl sequence and write overlays and video."""
    seq_name = jsonl_path.stem  # e.g., 2022-07-14--14-32-55--10_first
    seq_out_dir = out_root / seq_name
    seq_out_dir.mkdir(parents=True, exist_ok=True)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = out_root / f"{seq_name}.mp4"
    vw = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    # First, load all frames into memory for future trajectory lookup
    frames_data = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Loading {seq_name}"):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            frames_data.append(data)

    print(f"Loaded {len(frames_data)} frames for {seq_name}")

    # Process each frame with future trajectories
    for current_idx, current_data in enumerate(tqdm(frames_data, desc=f"Processing {seq_name}")):
        # Load current image
        rel_img = current_data["image_path"]  # relative path stored inside JSON
        img_path = dataset_root / rel_img
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue

        # Ensure size
        if (img.shape[1], img.shape[0]) != (width, height):
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

        # Current frame matrices
        K = np.array(current_data["intrinsic_matrix"], dtype=np.float64)
        T = np.array(current_data["extrinsic_matrix"], dtype=np.float64)

        # Current trajectory with height adjustment for road surface projection
        current_traj = np.array(current_data["trajectory"], dtype=np.float64)  # (N,3)
        
        # Apply height correction: trajectory coordinates are relative to vehicle base,
        # but we want to project onto the road surface which is ~0.6m above vehicle reference
        if len(current_traj) > 0:
            current_traj_adjusted = current_traj.copy()
            current_traj_adjusted[:, 2] += 0.6  # Elevate trajectory points to road surface level
        
        # Transform current trajectory to camera and project
        if len(current_traj) > 0:
            pts_cam = world_to_camera(current_traj_adjusted, T)
            # Filter points that are in front of the camera (positive Z) and not too close
            mask_valid = (pts_cam[:, 2] > 0.1) & (pts_cam[:, 2] < 100.0)  # Between 10cm and 100m
            
            current_traj_uv = None
            if np.any(mask_valid):
                pts_cam_valid = pts_cam[mask_valid]
                projected_uv = project_points(pts_cam_valid, K)
                
                # Filter points that project within reasonable image bounds (including some margin)
                h, w = img.shape[:2]
                margin = 100  # Allow some points outside immediate view
                mask_in_bounds = (
                    (projected_uv[:, 0] >= -margin) & (projected_uv[:, 0] <= w + margin) &
                    (projected_uv[:, 1] >= -margin) & (projected_uv[:, 1] <= h + margin)
                )
                
                if np.any(mask_in_bounds):
                    current_traj_uv = projected_uv[mask_in_bounds]

        # Get only the first future trajectory (next step ahead)
        future_traj_uv = None
        if current_idx + 1 < len(frames_data):
            future_idx = current_idx + 1
            future_data = frames_data[future_idx]
            
            # Get future trajectory in world coordinates with height adjustment
            future_traj = np.array(future_data["trajectory"], dtype=np.float64)
            
            if len(future_traj) > 0:
                # Apply same height correction for road surface projection
                future_traj_adjusted = future_traj.copy()
                future_traj_adjusted[:, 2] += 0.6  # Elevate to road surface level
                
                # Transform using current frame's camera matrices (to show where future trajectory would appear in current view)
                future_pts_cam = world_to_camera(future_traj_adjusted, T)
                future_mask_valid = (future_pts_cam[:, 2] > 0.1) & (future_pts_cam[:, 2] < 100.0)
                
                if np.any(future_mask_valid):
                    future_pts_cam_valid = future_pts_cam[future_mask_valid]
                    future_projected_uv = project_points(future_pts_cam_valid, K)
                    
                    # Filter points within reasonable bounds
                    h, w = img.shape[:2]
                    margin = 100
                    future_mask_in_bounds = (
                        (future_projected_uv[:, 0] >= -margin) & (future_projected_uv[:, 0] <= w + margin) &
                        (future_projected_uv[:, 1] >= -margin) & (future_projected_uv[:, 1] <= h + margin)
                    )
                    
                    if np.any(future_mask_in_bounds):
                        future_traj_uv = future_projected_uv[future_mask_in_bounds]

        # Create overlay with current trajectory only
        overlay = img.copy()
        
        # Draw current trajectory as connected green line with waypoints
        if current_traj_uv is not None and current_traj_uv.size > 0:
            # Draw connecting line in green
            draw_trajectory(overlay, current_traj_uv, color=(0, 255, 0), thickness=2)
            # Draw current trajectory waypoints as green dots
            draw_trajectory_waypoints(overlay, current_traj_uv, color=(0, 255, 0), radius=5)  # Bigger green dots
            # Add a larger blue dot at the start of current trajectory (vehicle position)
            if len(current_traj_uv) > 0:
                p0 = tuple(np.round(current_traj_uv[0]).astype(int))
                cv2.circle(overlay, p0, 10, (255, 0, 0), -1, cv2.LINE_AA)  # Bigger blue dot for vehicle

        # Add simplified caption in bottom left corner
        speed = current_data.get("ego_state", {}).get("vEgo", None)
        
        # Position text in bottom left corner
        caption_y = overlay.shape[0] - 60  # 60 pixels from bottom
        
        # Updated caption for green trajectory only
        caption_text = "trajectory in green"
        speed_text = f"Speed: {speed:.2f} m/s" if speed is not None else "Speed: N/A"
        
        # Larger font size for better visibility
        cv2.putText(overlay, caption_text, (20, caption_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, speed_text, (20, caption_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

        # Save and write to video
        out_path = seq_out_dir / f"{Path(rel_img).stem}.png"
        cv2.imwrite(str(out_path), overlay)
        vw.write(overlay)

    vw.release()
    print(f"Saved: {video_path}")


def find_sequences(dataset_root: Path) -> List[Path]:
    states_dir = dataset_root / "states"
    return sorted(states_dir.glob("*.jsonl"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Project CoVLA trajectories onto images and videos.")
    ap.add_argument("--dataset-root", type=str, required=True, help="Path to dataset root (contains images/, states/, video_samples/)")
    ap.add_argument("--fps", type=int, default=20, help="Output video FPS (default 20)")
    ap.add_argument("--width", type=int, default=1928, help="Image width (default 1928)")
    ap.add_argument("--height", type=int, default=1208, help="Image height (default 1208)")
    ap.add_argument("--only-seq", type=str, default=None, help="Optional sequence stem to process (e.g. 2022-07-14--14-32-55--10_first)")
    ap.add_argument("--num-future-trajectories", type=int, default=20, help="Number of future trajectories to project (default 20)")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    out_root = dataset_root / "overlays"
    out_root.mkdir(parents=True, exist_ok=True)

    jsonl_files = find_sequences(dataset_root)
    if args.only_seq:
        jsonl_files = [p for p in jsonl_files if p.stem == args.only_seq]
        if not jsonl_files:
            raise SystemExit(f"No sequence named {args.only_seq} under {dataset_root/'states'}")

    if not jsonl_files:
        raise SystemExit(f"No JSONL files found under {dataset_root/'states'}")

    for p in jsonl_files:
        process_sequence(dataset_root, p, out_root, args.width, args.height, args.fps, args.num_future_trajectories)


if __name__ == "__main__":
    main()
