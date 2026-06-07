# RobotCutout Web Demo

This repository packages a browser-based RobotSeg cutout workflow. It can run RobotSeg on a folder of robot images, open a local Flask web UI, and let you refine the predicted mask with brush, pen, zoom, and move tools before saving transparent cutouts.

The original upstream README has been backed up as `README.original.md`.

## 1. Environment Setup

RobotSeg expects a GPU Python environment. The upstream project recommends Python 3.11 with PyTorch 2.5.1 and torchvision 0.20.1.

### Create Conda Environment

```powershell
conda create -n robotseg python=3.11
conda activate robotseg
```

### Install PyTorch

For CUDA 12.1:

```powershell
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

If your CUDA version is different, install the matching PyTorch build from the official PyTorch selector.

### Install Project Dependencies

From the repository root:

```powershell
pip install -e ".[dev]"
python setup.py build_ext --inplace
```

The `dev` extra includes packages used by the demo, such as Flask, OpenCV, natsort, matplotlib, and related utilities.

If `cv2.ximgproc.guidedFilter` is unavailable in your OpenCV build, the demo now falls back safely. For better guided-filter refinement, install an OpenCV contrib build:

```powershell
pip install opencv-contrib-python-headless==4.11.0.86
```

## 2. Download Model Weights

Download `robotseg.pt` and place it in the `checkpoints` folder:

```text
checkpoints/robotseg.pt
```

Available download links from the original RobotSeg README:

- OneDrive: https://1drv.ms/u/c/f6d9d790b8550d3f/IQDc3mfIAQRETb7zmyhO-BG5AU-cIxzPnUwBDlsrCgcEQ3k?e=oT7NtR
- BaiduDisk: https://pan.baidu.com/s/1dkjD9YpFz4B2WcL2hkpUOA?pwd=cvpr

The default config file is already expected at:

```text
robotseg/configs/robotseg-infer.yaml
```

## 3. Prepare Input Images

By default, the Web demo reads images from:

```text
test/demo_video
```

Put your input images in that folder. Supported extensions are:

```text
.jpg, .jpeg, .png
```

You can also pass another folder with `--input_dir`.

## 4. Run the Cutout Web Demo

From the repository root:

```powershell
python test\demo_cutout_web.py
```

Then open your browser at:

```text
http://127.0.0.1:8080
```

Example with a custom input folder and port:

```powershell
python test\demo_cutout_web.py --input_dir H:\path\to\images --port 8080
```

If you want to disable guided filtering:

```powershell
python test\demo_cutout_web.py --guided_filter false
```

## 5. Web UI Usage

### Mask Mode

- `Add FG`: brush or pen edits add pixels to the cutout mask.
- `Remove BG`: brush or pen edits remove pixels from the cutout mask.

### Tools

- `Brush`: left-drag to paint mask edits.
- `Pen`: create and edit Bezier paths, then apply the path as a mask selection.
- `Move`: left-drag to move the image view only. It does not change the mask or pen path.

### View Controls

- `Ctrl + right mouse drag up`: zoom in around the mouse position.
- `Ctrl + right mouse drag down`: zoom out around the mouse position.
- `Move` tool + left mouse drag: pan the zoomed image.

### Common Actions

- `Ctrl + Z`: undo the previous edit.
- `Ctrl + K`, `Ctrl + Backspace`, or `Ctrl + Delete`: clear current mask.
- `Prev` / `Next`: switch frames.
- `Save`: save the current frame output.
- `Upload Images`: append new images from the browser and run inference on them.

## 6. Output Files

By default, outputs are saved under:

```text
test/demo_video_web_output/<category>/
```

Default subfolders:

```text
test/demo_video_web_output/robot/cutout
test/demo_video_web_output/robot/overlay
```

- `cutout`: transparent cutout result, PNG by default.
- `overlay`: preview image with mask overlay, JPG by default.

## 7. Command-Line Parameters

```text
--input_dir          Input image folder. Default: test/demo_video
--project_root       Project root containing robotseg/ and checkpoints/. Default: repository root
--output_root        Output root folder. Default: test/demo_video_web_output
--category           Segmentation target: arm, gripper, or robot. Default: robot
--checkpoint         Checkpoint name without .pt. Default: robotseg
--yaml               Config name without .yaml. Default: robotseg-infer
--guided_filter      Enable guided filter refinement. Default: true
--host               Flask host. Default: 127.0.0.1
--port               Flask port. Default: 8080
--max_frames         If greater than 0, only process the first N frames. Default: 0
--output_format      Cutout format: png, jpg, jpeg. Default: png
--overlay_format     Overlay format: jpg, jpeg, png, bmp, webp. Default: jpg
--cutout_dirname     Cutout output subfolder. Default: cutout
--overlay_dirname    Overlay output subfolder. Default: overlay
```

## 8. Troubleshooting

### No Images Found

If you see `No images found`, either create `test/demo_video` and put images inside it, or pass a custom folder:

```powershell
python test\demo_cutout_web.py --input_dir H:\path\to\images
```

### Missing Checkpoint

Make sure the weight file exists here:

```text
checkpoints/robotseg.pt
```

### OpenCV ximgproc Error

If your OpenCV package does not include `cv2.ximgproc.guidedFilter`, the demo will return the original mask for that refinement step. You can either keep using it as-is, run with `--guided_filter false`, or install OpenCV contrib:

```powershell
pip install opencv-contrib-python-headless==4.11.0.86
```

### CUDA or Torch Error

Confirm that PyTorch can see your GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If this prints `False`, reinstall PyTorch with the CUDA version that matches your machine.
