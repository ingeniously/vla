import numpy as np
import json

# Load test data
with open('C:/Users/USER/Pictures/BASEPIC/covla-mini/states/2022-07-14--14-32-55--10_first.jsonl', 'r') as f:
    line = f.readline()
    data = json.loads(line)


K = np.array(data['intrinsic_matrix'], dtype=np.float64)
T = np.array(data['extrinsic_matrix'], dtype=np.float64)
traj = np.array(data['trajectory'], dtype=np.float64)

def to_homogeneous(points_xyz):
    ones = np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)
    return np.concatenate([points_xyz, ones], axis=1)

def project_points(points_cam, K):
    x = points_cam[:, 0] / points_cam[:, 2]
    y = points_cam[:, 1] / points_cam[:, 2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * x + cx
    v = fy * y + cy
    return np.stack([u, v], axis=1)

# Test trajectory points at different heights
test_distances = [1, 2, 5, 10, 15, 20]  # meters ahead
target_y_range = (800, 1100)  # Y coordinates where trajectory should appear

print("Finding optimal trajectory height for road surface projection...")
print(f"Target Y range in image: {target_y_range}")
print()

best_z = None
best_error = float('inf')

# Try different Z offsets
for z_offset in np.arange(-2.0, 1.0, 0.1):  # Try heights from -2m to +1m
    errors = []
    for dist in test_distances:
        # Create test point at distance ahead
        test_point = np.array([[dist, 0.0, z_offset]])
        
        # Transform to camera
        pts_cam = (T @ to_homogeneous(test_point).T).T[:, :3]
        
        if pts_cam[0, 2] > 0.1:  # In front of camera
            projected = project_points(pts_cam, K)[0]
            y_proj = projected[1]
            
            # Check if Y is in reasonable range
            if target_y_range[0] <= y_proj <= target_y_range[1]:
                error = abs(y_proj - (target_y_range[0] + target_y_range[1]) / 2)
                errors.append(error)
    
    if errors:
        avg_error = np.mean(errors)
        if avg_error < best_error:
            best_error = avg_error
            best_z = z_offset

print(f"Best Z offset: {best_z:.2f}m")
print()

# Test with best Z offset
print("Testing trajectory projection with optimal height:")
for i, dist in enumerate(test_distances):
    test_point = np.array([[dist, 0.0, best_z]])
    pts_cam = (T @ to_homogeneous(test_point).T).T[:, :3]
    
    if pts_cam[0, 2] > 0.1:
        projected = project_points(pts_cam, K)[0]
        print(f"{dist:2d}m ahead: X={projected[0]:7.1f}, Y={projected[1]:7.1f}")

print()
print("Now testing actual trajectory with height adjustment...")

# Apply to actual trajectory points
traj_adjusted = traj.copy()
traj_adjusted[:, 2] = best_z  # Set all Z coordinates to optimal height

# Transform first few points
test_points = traj_adjusted[:8]
pts_cam = (T @ to_homogeneous(test_points).T).T[:, :3]
valid_mask = pts_cam[:, 2] > 0.1

if np.any(valid_mask):
    projected = project_points(pts_cam[valid_mask], K)
    print("Adjusted trajectory projection (first 8 points):")
    for i, proj in enumerate(projected):
        print(f"Point {i}: X={proj[0]:7.1f}, Y={proj[1]:7.1f}")