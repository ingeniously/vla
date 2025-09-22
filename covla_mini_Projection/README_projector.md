# CoVLA Trajectory Projection Toolkit

Complete pipeline to extract, flatten, and project CoVLA 3D trajectories onto images and videos.


## Features
- Extract .tar.gz archives with nested directory structures
- Flatten complex nested folders automatically
- Project vehicle trajectories onto camera images with proper 3D-to-2D transformation
- Generate annotated videos showing current trajectory in green with vehicle position in blue
- Height-corrected trajectory projection for accurate road surface visualization

## Requirements
- Packages: numpy, opencv-python (or opencv-contrib-python), tqdm

## Setup 

```bash
# Go to your dataset root
cd /home/.../vla/covla-mini
```

## Complete Workflow

### Step 1: Extract Archived Images (if images/ contains .tar.gz)
```bash
# Extract all archives and flatten nested structures into images/<sequence>/*.png
python /home/../vla/vila-u/covla_mini_Projection/extract_archives.py \ --src "/home/../vla/covla-mini/images"
```

Output structure example:
```
/home/../vla/covla-mini/images/
├── 2022-07-14--14-32-55--10_first/
│   ├── 0000.png
│   ├── 0001.png
│   └── ...
├── 2022-07-14--14-32-55--11_first/
│   ├── 0000.png
│   ├── 0001.png
│   └── ...
```


### Step 2: Project Trajectories
Project 3D trajectories onto images and export overlays + video.

Process all sequences:
```bash
python /home/../vla/vila-u/covla_mini_Projection/project_trajectories.py \
  --dataset-root "/home/../covla-mini" \
  --fps 20 --width 1928 --height 1208
```

Process a single sequence (faster for testing):
```bash
python /home/../vla/covla_mini_Projection/project_trajectories.py \
  --dataset-root "/home/../covla-mini" \
  --only-seq "2022-07-14--14-32-55--10_first" \
  --fps 20 --width 1928 --height 1208
```

## Output

- Annotated images:/home/../vla/covla-mini/overlays/<sequence>/<frame>.png
- Video per sequence: /home/../vla/covla-mini/overlays/<sequence>.mp4

### Visualization Elements
- Green trajectory: current vehicle path with waypoints (thick dots)
- Blue dot: vehicle’s current position
- Caption: “trajectory in green” + speed in m/s
- Height correction: +0.6 m applied for road-surface projection

## Notes 
- If your python is not python3, replace commands accordingly.
- If scripts are not executable, you can run them explicitly with python as shown.
- Ensure images/, states/, and (optional) video_samples/ exist at //home/../vla/covla-mini.