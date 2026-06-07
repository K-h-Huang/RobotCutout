# -*- coding: utf-8 -*-
# @FileName: demo_cutout_web.py
# Web interactive demo for RobotSeg cutout (browser-based editing)

import argparse
import base64
import os
import time
from pathlib import Path
import cv2
import numpy as np

try:
    from natsort import natsorted
except Exception:
    natsorted = sorted

try:
    from flask import Flask, jsonify, request
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Flask is required for web mode. Install with: pip install flask"
    ) from exc

try:
    from test.utils import *
except Exception:
    from utils import *

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = str(SCRIPT_DIR / "demo_video")
DEFAULT_OUTPUT_ROOT = str(SCRIPT_DIR / "demo_video_web_output")
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parent
CATEGORY = "robot"
CKPT_NAME = "robotseg"
YAML_NAME = "robotseg-infer"


def get_image_list(input_dir):
    input_dir = Path(input_dir)
    exts = ["*.jpg", "*.jpeg", "*.png"]
    image_list = []
    seen = set()
    for ext in exts:
        for p in input_dir.glob(ext):
            s = str(p)
            if s not in seen:
                image_list.append(s)
                seen.add(s)
    return natsorted(image_list)


def get_data_url(img_bgr, ext="jpg", include_alpha=False):
    if include_alpha and img_bgr is not None and img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        ok, buf = cv2.imencode(".png", img_bgr)
        mime = "image/png"
    else:
        if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
        ok, buf = cv2.imencode(f".{ext}", img_bgr)
        mime = "image/jpeg" if ext == "jpg" else f"image/{ext}"
    if not ok:
        raise RuntimeError(f"Failed encoding image for API response: ext={ext}")
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Web interactive RobotSeg cutout demo (browser-based editing)."
    )
    parser.add_argument("--input_dir", default=INPUT_DIR, help="Directory that contains image frames.")
    parser.add_argument(
        "--project_root",
        default=str(DEFAULT_PROJECT_ROOT),
        help="Project root containing robotseg/, checkpoints/, and config/model files.",
    )
    parser.add_argument(
        "--output_root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory to save overlay and cutout outputs.",
    )
    parser.add_argument("--category", default=CATEGORY, choices=["arm", "gripper", "robot"])
    parser.add_argument("--checkpoint", default=CKPT_NAME, help="Checkpoint file name without suffix.")
    parser.add_argument("--yaml", default=YAML_NAME, help="RobotSeg config name, without extension.")
    parser.add_argument(
        "--guided_filter",
        type=lambda x: str(x).lower() in ["true", "1", "yes", "y", "on"],
        default=True,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Flask host.")
    parser.add_argument("--port", type=int, default=8080, help="Flask port.")
    parser.add_argument("--max_frames", type=int, default=0, help="If >0, only process first N frames.")
    parser.add_argument(
        "--output_format",
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Output format for cutout save (overlay always saved as image format).",
    )
    parser.add_argument(
        "--overlay_format",
        default="jpg",
        choices=["jpg", "jpeg", "png", "bmp", "webp"],
    )
    parser.add_argument(
        "--cutout_dirname",
        default="cutout",
        help="Subfolder name for RGBA cutouts under output_root/category.",
    )
    parser.add_argument(
        "--overlay_dirname",
        default="overlay",
        help="Subfolder name for overlay results under output_root/category.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_root = Path(args.output_root).resolve()
    project_root = Path(args.project_root).resolve()

    image_list = get_image_list(input_dir)
    if len(image_list) == 0:
        raise FileNotFoundError(f"No images found in {input_dir}")
    if args.max_frames > 0:
        image_list = image_list[: args.max_frames]

    overlay_dir = output_root / args.category / args.overlay_dirname
    cutout_dir = output_root / args.category / args.cutout_dirname
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(cutout_dir, exist_ok=True)

    model_cfg = str((project_root / "robotseg" / "configs" / f"{args.yaml}.yaml").resolve())
    checkpoint = str((project_root / "checkpoints" / f"{args.checkpoint}.pt").resolve())
    if not (Path(model_cfg).exists() and Path(checkpoint).exists()):
        # fallback to test parent, which is common for current workspace layout
        fallback_root = Path(__file__).resolve().parent.parent
        model_cfg_fb = str((fallback_root / "robotseg" / "configs" / f"{args.yaml}.yaml").resolve())
        checkpoint_fb = str((fallback_root / "checkpoints" / f"{args.checkpoint}.pt").resolve())
        if Path(model_cfg_fb).exists() and Path(checkpoint_fb).exists():
            model_cfg, checkpoint, project_root = model_cfg_fb, checkpoint_fb, fallback_root
        else:
            raise FileNotFoundError(f"Config/checkpoint not found under: {project_root}")

    import torch
    from robotseg.build_robotseg import build_robotseg_video_predictor

    torch.cuda.set_device(0)
    predictor = build_robotseg_video_predictor(model_cfg, checkpoint)

    # Run inference once
    print("Running inference...")
    results = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(
            video_path=str(input_dir),
            async_loading_frames=False,
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
        )
        predictor.add_new_robot(
            inference_state=state,
            frame_idx=0,
            obj_id=0,
            robot=args.category,
        )
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state=state,
            robot=args.category,
        ):
            results[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).detach().cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

    # Build working masks and image cache
    frame_data = []
    for frame_idx, image_path in enumerate(image_list):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        if frame_idx in results and 0 in results[frame_idx]:
            pred_mask = np.squeeze(results[frame_idx][0])
            pred_mask = (pred_mask > 0).astype(np.uint8) * 255
            if args.guided_filter:
                pred_mask = guided_refine_mask(pred_mask, image)
        else:
            pred_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        pred_mask = cv2.resize(pred_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        frame_data.append(
            {
                "image_path": image_path,
                "image_name": Path(image_path).name,
                "image": image,
                "base_mask": pred_mask.copy(),
                "mask": pred_mask.copy(),
            }
        )

    def infer_masks_for_images(video_dir, images):
        """Run RobotSeg on a folder-backed batch and return masks aligned to images."""
        if len(images) == 0:
            return []
        upload_results = {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(
                video_path=str(video_dir),
                async_loading_frames=False,
                offload_video_to_cpu=False,
                offload_state_to_cpu=False,
            )
            predictor.add_new_robot(
                inference_state=state,
                frame_idx=0,
                obj_id=0,
                robot=args.category,
            )
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state=state,
                robot=args.category,
            ):
                upload_results[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).detach().cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

        masks = []
        for idx, image in enumerate(images):
            if idx in upload_results and 0 in upload_results[idx]:
                pred_mask = np.squeeze(upload_results[idx][0])
                pred_mask = (pred_mask > 0).astype(np.uint8) * 255
                if args.guided_filter:
                    pred_mask = guided_refine_mask(pred_mask, image)
            else:
                pred_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            pred_mask = cv2.resize(pred_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            masks.append(pred_mask)
        return masks

    # Flask app
    app = Flask(__name__)

    @app.route("/")
    def index():
        return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>RobotSeg Web Cutout</title>
    <style>
    :root {
      --bg: #eef2ef;
      --panel: rgba(255,255,250,0.88);
      --panel-strong: #fffdf5;
      --ink: #18211f;
      --muted: #64716c;
      --line: rgba(24,33,31,0.13);
      --accent: #0f766e;
      --accent-2: #d97706;
      --danger: #b91c1c;
      --shadow: 0 20px 55px rgba(24,33,31,0.14);
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      height: 100vh;
      margin: 0;
      overflow: auto;
      color: var(--ink);
      font-family: "Aptos", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 12% 8%, rgba(15,118,110,0.18), transparent 28%),
        radial-gradient(circle at 92% 12%, rgba(217,119,6,0.16), transparent 24%),
        linear-gradient(135deg, #f8f7ef 0%, var(--bg) 46%, #e6ece8 100%);
    }
    .app-shell {
      display: grid;
      grid-template-columns: minmax(195px, 250px) minmax(0, 1fr);
      gap: 4px;
      height: 100vh;
      padding: 7px;
    }
    .toolbar {
      align-self: start;
      position: sticky;
      top: 10px;
      display: flex;
      flex-direction: column;
      gap: 4px;
      max-height: calc(100vh - 20px);
      overflow: auto;
      padding: 7px;
      border: 1px solid rgba(255,255,255,0.75);
      border-radius: 16px;
      background: var(--panel);
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
    }
    .brand {
      display: grid;
      gap: 2px;
      padding-bottom: 6px;
      border-bottom: 1px solid var(--line);
    }
    .brand h1 {
      margin: 0;
      font-size: 14px;
      letter-spacing: -0.04em;
    }
    .brand p {
      margin: 0;
      color: var(--muted);
      font-size: 9px;
      line-height: 1.18;
    }
    .frame-strip {
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 4px;
      align-items: center;
    }
    .frame-pill {
      display: inline-flex;
      align-items: baseline;
      gap: 4px;
      min-height: 22px;
      padding: 4px 6px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255,255,255,0.62);
      font-weight: 700;
      font-size: 10px;
    }
    .frame-pill span:last-child, .frame-pill .slash { color: var(--muted); font-weight: 600; }
    .group {
      display: grid;
      gap: 4px;
      padding: 6px;
      border: 1px solid var(--line);
      border-radius: 13px;
      background: rgba(255,255,255,0.54);
    }
    .group-title {
      color: var(--muted);
      font-size: 9px;
      font-weight: 800;
      letter-spacing: 0.1em;
      text-transform: uppercase;
    }
    .button-row {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
    }
    button {
      min-height: 22px;
      border: 1px solid rgba(24,33,31,0.14);
      border-radius: 999px;
      padding: 4px 6px;
      color: var(--ink);
      background: linear-gradient(180deg, #ffffff 0%, #f3f1e8 100%);
      box-shadow: 0 1px 0 rgba(255,255,255,0.8) inset;
      cursor: pointer;
      font-weight: 700;
      font-size: 10px;
      transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease, background 120ms ease;
    }
    button:hover { transform: translateY(-1px); box-shadow: 0 8px 18px rgba(24,33,31,0.12); }
    button:active { transform: translateY(0); }
    button.active, .toolbar .active {
      color: #fff;
      border-color: transparent;
      background: linear-gradient(135deg, var(--accent) 0%, #0b4f49 100%);
      box-shadow: 0 10px 22px rgba(15,118,110,0.28);
    }
    button[onclick="saveFrame()"] {
      color: #fff;
      border-color: transparent;
      background: linear-gradient(135deg, #111827 0%, #26332f 100%);
    }
    button[onclick="clearMaskCanvas()"] { color: var(--danger); }
    .upload-control,
    .brush-control {
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 9px;
      font-weight: 700;
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }
    .upload-control input[type="file"] {
      width: 100%;
      font-size: 9px;
    }
    #brushSizePanel,
    #penPathPanel { display: none; }
    #brushSizePanel.visible,
    #penPathPanel.visible { display: grid; }
    #penCount {
      align-self: center;
      justify-self: start;
      padding: 4px 6px;
      border-radius: 999px;
      color: var(--accent);
      background: rgba(15,118,110,0.09);
      font-size: 9px;
      font-weight: 800;
    }
    #status { display: none; }
    .help-card {
      display: grid;
      gap: 4px;
      padding: 7px;
      border: 1px solid rgba(15,118,110,0.18);
      border-radius: 13px;
      background: linear-gradient(180deg, rgba(255,255,255,0.72), rgba(239,247,243,0.72));
    }
    .help-card strong {
      font-size: 9px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent);
    }
    #shortcutHelp {
      margin: 0;
      padding-left: 12px;
      color: #38514b;
      font-size: 9px;
      line-height: 1.18;
    }
    #shortcutHelp li { margin: 1px 0; }
    .canvas-wrap {
      min-width: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      height: calc(100vh - 24px);
      overflow: auto;
      padding: 7px;
      border: 1px solid rgba(255,255,255,0.72);
      border-radius: 22px;
      background:
        linear-gradient(45deg, rgba(24,33,31,0.035) 25%, transparent 25%),
        linear-gradient(-45deg, rgba(24,33,31,0.035) 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, rgba(24,33,31,0.035) 75%),
        linear-gradient(-45deg, transparent 75%, rgba(24,33,31,0.035) 75%),
        rgba(255,255,250,0.58);
      background-size: 24px 24px;
      background-position: 0 0, 0 12px, 12px -12px, -12px 0;
      box-shadow: var(--shadow);
    }
    #stage {
      position: relative;
      display: inline-block;
      flex: 0 0 auto;
      overflow: hidden;
      border: 1px solid rgba(24,33,31,0.2);
      border-radius: 13px;
      background: #101513;
      box-shadow: 0 18px 50px rgba(0,0,0,0.18);
    }
    #baseImage { display: block; max-width: none; }
    #overlayCanvas { position: absolute; left: 0; top: 0; cursor: crosshair; touch-action: none; }
    #penSvg { position: absolute; left: 0; top: 0; pointer-events: none; overflow: visible; }
    #penContextMenu {
      position: fixed;
      display: none;
      z-index: 1000;
      min-width: 170px;
      padding: 6px;
      border: 1px solid rgba(24,33,31,0.16);
      border-radius: 14px;
      background: var(--panel-strong);
      box-shadow: 0 18px 45px rgba(24,33,31,0.2);
    }
    #penContextMenu button {
      display: block;
      width: 100%;
      border: 0;
      border-radius: 10px;
      background: transparent;
      box-shadow: none;
      padding: 9px 10px;
      text-align: left;
      cursor: pointer;
    }
    #penContextMenu button:hover { background: rgba(15,118,110,0.09); transform: none; box-shadow: none; }
    @media (max-width: 560px) {
      body { overflow: auto; }
      .app-shell { grid-template-columns: 1fr; height: auto; min-height: 100vh; padding: 10px; }
      .toolbar { position: static; max-height: none; }
      .frame-strip { grid-template-columns: 1fr 1fr; }
      .frame-pill { grid-column: 1 / -1; }
      .canvas-wrap { height: min(68vh, 720px); }
    }
  </style>
</head>
<body>
<div class="app-shell">
  <aside class="toolbar">
    <div class="brand">
      <h1>RobotSeg Cutout</h1>
      <p>Brush and pen refinements share one mask. Save exports the current cutout.</p>
    </div>

    <div class="frame-strip">
      <div class="frame-pill">Frame <span id="frameIdx">0</span><span class="slash">/</span><span id="frameCount">0</span></div>
      <button onclick="prevFrame()">Prev</button>
      <button onclick="nextFrame()">Next</button>
    </div>

    <div class="group">
      <div class="group-title">Input</div>
      <label class="upload-control">Upload Images <input id="uploadImages" type="file" accept="image/*" multiple/></label>
    </div>

    <div class="group">
      <div class="group-title">Mask Mode</div>
      <div class="button-row">
        <button id="modeFgBtn" onclick="setMode('fg')">Add FG</button>
        <button id="modeBgBtn" onclick="setMode('bg')">Remove BG</button>
      </div>
    </div>

    <div class="group">
      <div class="group-title">Tool</div>
      <div class="button-row">
        <button id="toolBrushBtn" onclick="setTool('brush')">Brush</button>
        <button id="toolPenBtn" onclick="setTool('pen')">Pen</button>
      </div>
      <label id="brushSizePanel" class="brush-control">Brush Size <input id="brush" type="range" min="2" max="80" value="20"/></label>
    </div>

    <div id="penPathPanel" class="group">
      <div class="group-title">Pen Path</div>
      <div class="button-row">
        <button onclick="closePenPath()" title="C">Close Path</button>
        <button onclick="applyPenToMask()" title="Enter">Make Selection</button>
        <button onclick="clearPenPath()" title="Esc">Clear Pen</button>
      </div>
      <span id="penCount">Pts: 0</span>
    </div>

    <div class="group">
      <div class="group-title">Mask Actions</div>
      <div class="button-row">
        <button onclick="undoEdit()" title="Ctrl+Z">Undo</button>
        <button onclick="clearMaskCanvas()" title="Ctrl+K / Ctrl+Backspace / Ctrl+Delete">Clear Mask</button>
        <button onclick="resetFrame()">Reset</button>
        <button onclick="saveFrame()">Save</button>
      </div>
    </div>

    <span id="status" aria-hidden="true"></span>

    <div class="help-card">
      <strong>操作说明</strong>
      <ul id="shortcutHelp"></ul>
    </div>
  </aside>

  <main class="canvas-wrap">
    <div id="stage">
      <img id="baseImage" />
      <canvas id="overlayCanvas"></canvas>
      <svg id="penSvg" aria-hidden="true"></svg>
    </div>
  </main>
</div>
<canvas id="maskCanvas" style="display:none;"></canvas>
<div id="penContextMenu"><button id="penCurveToggleBtn" type="button" onclick="toggleContextAnchorCurve()">Convert to Curve</button><button type="button" onclick="deleteContextAnchor()">Delete Anchor</button></div>
<script>
let currentFrame = 0;
let totalFrames = 0;
let maskDirty = false;
let mode = "fg";
let tool = "brush";
let isMouseDown = false;
let brushDirty = false;
let dragObject = null;
let selectedPenObject = null;
let contextAnchorIndex = -1;
let penPoints = [];
let isPenClosed = false;
let lastPointer = null;
let brushPreview = null;
let imageFitScale = 1;
let imageScaleFactor = 1;
let resizeDrag = null;
let resizeRaf = null;
let undoStack = [];
const brushInput = document.getElementById("brush");
const uploadImagesInput = document.getElementById("uploadImages");
const baseImage = document.getElementById("baseImage");
const overlayCanvas = document.getElementById("overlayCanvas");
const overlayCtx = overlayCanvas.getContext("2d");
const penSvg = document.getElementById("penSvg");
const maskCanvas = document.getElementById("maskCanvas");
const maskCtx = maskCanvas.getContext("2d");
const penContextMenu = document.getElementById("penContextMenu");
const penCurveToggleBtn = document.getElementById("penCurveToggleBtn");
const hitRadius = 10;
const handleRadius = 8;
const segmentHitRadius = 7;
const dragThreshold = 3;
const sampleSteps = 36;
const resizeHitSize = 14;

function clonePoint(p){
  return {
    x: p.x,
    y: p.y,
    inHandle: { x: p.inHandle.x, y: p.inHandle.y },
    outHandle: { x: p.outHandle.x, y: p.outHandle.y },
    smooth: !!p.smooth
  };
}

function createAnchor(x, y){
  return {
    x,
    y,
    inHandle: { x, y },
    outHandle: { x, y },
    smooth: false
  };
}

function dist2(a, b){
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
}

function isPoint(obj, x, y, threshold){
  return dist2(obj, { x, y }) <= threshold * threshold;
}

function reflectHandle(anchor, movedHandle, targetHandle){
  targetHandle.x = anchor.x * 2 - movedHandle.x;
  targetHandle.y = anchor.y * 2 - movedHandle.y;
}

function handlesCollapsed(p){
  return isPoint(p.inHandle, p.x, p.y, 0.75) && isPoint(p.outHandle, p.x, p.y, 0.75);
}

function pushUndo(){
  if(maskCanvas.width === 0 || maskCanvas.height === 0){
    return;
  }
  undoStack.push({
    mask: maskCanvas.toDataURL("image/png"),
    points: penPoints.map(clonePoint),
    closed: isPenClosed
  });
  if(undoStack.length > 80){
    undoStack.shift();
  }
}

function restoreMaskFromUrl(maskUrl, afterLoad){
  const img = new Image();
  img.onload = () => {
    maskCtx.globalCompositeOperation = "source-over";
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    maskCtx.drawImage(img, 0, 0, maskCanvas.width, maskCanvas.height);
    if(afterLoad){
      afterLoad();
    }
    redrawOverlay();
    maskDirty = false;
  };
  img.src = maskUrl;
}

function undoEdit(){
  const previous = undoStack.pop();
  if(!previous){
    document.getElementById("status").textContent = "Nothing to undo.";
    return;
  }
  penPoints = previous.points.map(clonePoint);
  isPenClosed = previous.closed;
  dragObject = null;
  lastPointer = null;
  restoreMaskFromUrl(previous.mask, () => {
    document.getElementById("status").textContent = "Undo.";
    updateToolButtons();
  });
}

function updateModeButtons(){
  document.getElementById("modeFgBtn").classList.toggle("active", mode === "fg");
  document.getElementById("modeBgBtn").classList.toggle("active", mode === "bg");
  updateShortcutHelp();
}

function updateShortcutHelp(){
  const help = document.getElementById("shortcutHelp");
  if(!help){
    return;
  }
  const modeText = mode === "fg" ? "当前模式：Add FG，笔刷/钢笔会把区域加入选区。" : "当前模式：Remove BG，笔刷会擦除 mask，钢笔应用后会从选区扣除。";
  const common = [
    modeText,
    "Ctrl+Z：撤销上一步编辑。",
    "Ctrl+K / Ctrl+Backspace / Ctrl+Delete：清空当前 mask。",
    "Save：保存当前帧的 overlay 和 cutout。"
  ];
  const brushTips = [
    "Brush：鼠标圆圈显示笔刷大小，拖动即可涂抹。",
    "Brush Size：调节笔刷预览圆和实际涂抹半径。",
    "Upload Images：可从浏览器上传新图片，上传后会追加到帧列表。",
    "Prev / Next：切换帧前会自动合并并同步当前笔刷和钢笔编辑。"
  ];
  const penTips = [
    "Pen：单击空白处添加直线锚点，按住拖动新锚点拉出贝塞尔手柄。",
    "拖动锚点移动位置，拖动手柄调整曲线弧度。",
    "右键锚点/手柄：直线/曲线切换或删除锚点。",
    "C：闭合路径；Enter：把闭合路径生成选区并合并到 mask；Esc：清空当前钢笔路径。"
  ];
  const tips = (tool === "pen" ? penTips : brushTips).concat(common);
  help.innerHTML = tips.map(t => `<li>${t}</li>`).join("");
}

function updateToolButtons(){
  document.getElementById("toolBrushBtn").classList.toggle("active", tool === "brush");
  document.getElementById("toolPenBtn").classList.toggle("active", tool === "pen");
  document.getElementById("penCount").textContent = "Pts: " + penPoints.length + (isPenClosed ? " (closed)" : "");
  document.getElementById("brushSizePanel").classList.toggle("visible", tool === "brush");
  document.getElementById("penPathPanel").classList.toggle("visible", tool === "pen");
  overlayCanvas.style.cursor = tool === "pen" ? "crosshair" : (tool === "brush" ? "none" : "default");
  if(tool === "pen"){
    overlayCanvas.title = "Pen: click empty canvas to add a corner, click-drag while adding for Bezier handles, drag anchors/handles to edit, right-click an anchor/handle to toggle straight/curve or delete it, click the first anchor to close, Enter to merge.";
  } else {
    overlayCanvas.title = "Brush: paint foreground/background edits into the mask. The cursor circle shows the brush size.";
  }
  updateShortcutHelp();
}

function setMode(m){
  mode = m;
  updateModeButtons();
  redrawOverlay();
}

function setTool(t){
  tool = t;
  dragObject = null;
  selectedPenObject = null;
  if(tool !== "brush"){
    brushPreview = null;
  }
  updateToolButtons();
  redrawOverlay();
}

async function loadFrame(i, skipSync=false){
  if(!skipSync){
    await syncCurrentMask();
  }
  const r = await fetch(`/api/frame/${i}`).then(r=>r.json());
  totalFrames = r.total_frames;
  currentFrame = r.index;
  undoStack = [];
  document.getElementById("frameIdx").textContent = String(currentFrame + 1);
  document.getElementById("frameCount").textContent = String(totalFrames);

  baseImage.onload = () => {
    overlayCanvas.width = baseImage.naturalWidth;
    overlayCanvas.height = baseImage.naturalHeight;
    maskCanvas.width = baseImage.naturalWidth;
    maskCanvas.height = baseImage.naturalHeight;
    updateSize();
    renderOverlayFromBaseMask(r.mask);
    maskDirty = false;
  };
  baseImage.src = r.image;
  clearPenPath(true);
}

function renderOverlayFromBaseMask(maskUrl){
  const img = new Image();
  img.onload = () => {
    maskCtx.globalCompositeOperation = "source-over";
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    maskCtx.drawImage(img, 0, 0, maskCanvas.width, maskCanvas.height);
    redrawOverlay();
    maskDirty = false;
  };
  img.src = maskUrl;
}

function redrawOverlayBase(){
  const w = maskCanvas.width, h = maskCanvas.height;
  if(w === 0 || h === 0){
    return;
  }
  const data = maskCtx.getImageData(0, 0, w, h).data;
  const out = overlayCtx.createImageData(w, h);
  const outd = out.data;
  for(let i = 0; i < data.length; i += 4){
    if(data[i] > 127){
      outd[i] = 255;
      outd[i + 1] = 0;
      outd[i + 2] = 0;
      outd[i + 3] = 120;
    } else {
      outd[i + 3] = 0;
    }
  }
  overlayCtx.globalCompositeOperation = "source-over";
  overlayCtx.putImageData(out, 0, 0);
}

function drawBrushPreview(){
  if(tool !== "brush" || !brushPreview){
    return;
  }
  const b = screenToImagePx(Number(brushInput.value));
  overlayCtx.save();
  overlayCtx.globalCompositeOperation = "source-over";
  overlayCtx.setLineDash([screenToImagePx(6), screenToImagePx(4)]);
  overlayCtx.lineWidth = screenToImagePx(2);
  overlayCtx.strokeStyle = mode === "fg" ? "rgba(255,255,255,0.95)" : "rgba(20,20,20,0.95)";
  overlayCtx.fillStyle = mode === "fg" ? "rgba(255,0,0,0.08)" : "rgba(0,0,0,0.08)";
  overlayCtx.beginPath();
  overlayCtx.arc(brushPreview.x, brushPreview.y, b, 0, Math.PI * 2);
  overlayCtx.fill();
  overlayCtx.stroke();
  overlayCtx.setLineDash([]);
  overlayCtx.lineWidth = screenToImagePx(1);
  overlayCtx.strokeStyle = mode === "fg" ? "rgba(185,28,28,0.95)" : "rgba(255,255,255,0.95)";
  overlayCtx.beginPath();
  overlayCtx.arc(brushPreview.x, brushPreview.y, b + screenToImagePx(1.5), 0, Math.PI * 2);
  overlayCtx.stroke();
  overlayCtx.restore();
}

function redrawOverlay(){
  overlayCtx.globalCompositeOperation = "source-over";
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  redrawOverlayBase();
  if(tool === "pen" || penPoints.length > 0){
    drawPenPath(false);
  }
  drawBrushPreview();
}

function applyBrush(x, y){
  const b = screenToImagePx(Number(brushInput.value));
  if(mode === "fg"){
    maskCtx.globalCompositeOperation = "source-over";
    maskCtx.fillStyle = "white";
  } else {
    maskCtx.globalCompositeOperation = "destination-out";
    maskCtx.fillStyle = "rgba(0,0,0,1)";
  }
  maskCtx.beginPath();
  maskCtx.arc(x, y, b, 0, Math.PI * 2);
  maskCtx.fill();
  maskCtx.globalCompositeOperation = "source-over";
  maskDirty = true;
  redrawOverlay();
}

function getPos(e){
  const rect = baseImage.getBoundingClientRect();
  const sx = baseImage.naturalWidth / rect.width;
  const sy = baseImage.naturalHeight / rect.height;
  return {
    x: (e.clientX - rect.left) * sx,
    y: (e.clientY - rect.top) * sy
  };
}

function getDisplayScale(){
  const rect = baseImage.getBoundingClientRect();
  if(!baseImage.naturalWidth || rect.width <= 0){
    return 1;
  }
  return rect.width / baseImage.naturalWidth;
}

function screenToImagePx(px){
  return px / Math.max(getDisplayScale(), 0.0001);
}

function getResizeHandle(e){
  if(!baseImage.naturalWidth || !baseImage.naturalHeight){
    return null;
  }
  const rect = baseImage.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  if(x < 0 || y < 0 || x > rect.width || y > rect.height){
    return null;
  }
  const nearRight = rect.width - x <= resizeHitSize;
  const nearBottom = rect.height - y <= resizeHitSize;
  if(nearRight && nearBottom){
    return "corner";
  }
  if(nearRight){
    return "right";
  }
  if(nearBottom){
    return "bottom";
  }
  return null;
}

function updateResizeCursor(e){
  if(isMouseDown || resizeDrag){
    return false;
  }
  const handle = getResizeHandle(e);
  if(!handle){
    return false;
  }
  overlayCanvas.style.cursor = handle === "right" ? "ew-resize" : (handle === "bottom" ? "ns-resize" : "nwse-resize");
  return true;
}

function beginImageResize(e, handle){
  const rect = baseImage.getBoundingClientRect();
  resizeDrag = {
    handle,
    startX: e.clientX,
    startY: e.clientY,
    startW: rect.width,
    startH: rect.height,
    startScaleFactor: imageScaleFactor
  };
  isMouseDown = true;
  brushPreview = null;
  hidePenContextMenu();
  window.addEventListener("mousemove", onResizeWindowMove);
  window.addEventListener("mouseup", onResizeWindowUp);
  e.preventDefault();
}

function scheduleResizeUpdate(){
  if(resizeRaf !== null){
    return;
  }
  resizeRaf = requestAnimationFrame(() => {
    resizeRaf = null;
    updateSize();
  });
}

function flushResizeUpdate(){
  if(resizeRaf !== null){
    cancelAnimationFrame(resizeRaf);
    resizeRaf = null;
  }
  updateSize();
}

function applyImageResize(e){
  if(!resizeDrag){
    return false;
  }
  const dx = e.clientX - resizeDrag.startX;
  const dy = e.clientY - resizeDrag.startY;
  const scaleFromW = (resizeDrag.startW + dx) / baseImage.naturalWidth;
  const scaleFromH = (resizeDrag.startH + dy) / baseImage.naturalHeight;
  let nextScale = scaleFromW;
  if(resizeDrag.handle === "bottom"){
    nextScale = scaleFromH;
  } else if(resizeDrag.handle === "corner"){
    const dominantDelta = Math.abs(dx / Math.max(resizeDrag.startW, 1)) >= Math.abs(dy / Math.max(resizeDrag.startH, 1)) ? dx / Math.max(resizeDrag.startW, 1) : dy / Math.max(resizeDrag.startH, 1);
    nextScale = (1 + dominantDelta) * (resizeDrag.startW / baseImage.naturalWidth);
  }
  imageScaleFactor = Math.max(0.1, Math.min(8, nextScale / Math.max(imageFitScale, 0.0001)));
  scheduleResizeUpdate();
  return true;
}

function cubicPoint(p0, c1, c2, p3, t){
  const mt = 1 - t;
  const mt2 = mt * mt;
  const t2 = t * t;
  return {
    x: mt2 * mt * p0.x + 3 * mt2 * t * c1.x + 3 * mt * t2 * c2.x + t2 * t * p3.x,
    y: mt2 * mt * p0.y + 3 * mt2 * t * c1.y + 3 * mt * t2 * c2.y + t2 * t * p3.y
  };
}

function getSegment(i){
  const n = penPoints.length;
  if(n < 2){
    return null;
  }
  if(!isPenClosed && i >= n - 1){
    return null;
  }
  const p0 = penPoints[i];
  const p3 = penPoints[(i + 1) % n];
  return { p0, c1: p0.outHandle, c2: p3.inHandle, p3 };
}

function buildBezierPath(ctx, points, closed){
  const n = points.length;
  if(n === 0){
    return;
  }
  ctx.moveTo(points[0].x, points[0].y);
  if(n === 1){
    return;
  }
  const lastSeg = closed ? n : n - 1;
  for(let i = 0; i < lastSeg; i++){
    const p0 = points[i];
    const p3 = points[(i + 1) % n];
    ctx.bezierCurveTo(p0.outHandle.x, p0.outHandle.y, p3.inHandle.x, p3.inHandle.y, p3.x, p3.y);
  }
}

function svgNode(tag, attrs){
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs || {}).forEach(([key, value]) => el.setAttribute(key, value));
  return el;
}

function buildSvgPath(points, closed){
  if(points.length === 0){
    return "";
  }
  let d = `M ${points[0].x} ${points[0].y}`;
  const lastSeg = closed ? points.length : points.length - 1;
  for(let i = 0; i < lastSeg; i++){
    const p0 = points[i];
    const p3 = points[(i + 1) % points.length];
    d += ` C ${p0.outHandle.x} ${p0.outHandle.y} ${p3.inHandle.x} ${p3.inHandle.y} ${p3.x} ${p3.y}`;
  }
  if(closed){
    d += " Z";
  }
  return d;
}

function appendSvgHandle(anchor, kind, color, selected){
  const h = kind === "out" ? anchor.outHandle : anchor.inHandle;
  if(isPoint(h, anchor.x, anchor.y, 0.75)){
    return;
  }
  penSvg.appendChild(svgNode("line", {
    x1: anchor.x,
    y1: anchor.y,
    x2: h.x,
    y2: h.y,
    stroke: color,
    "stroke-width": 1,
    "vector-effect": "non-scaling-stroke"
  }));
  penSvg.appendChild(svgNode("circle", {
    cx: h.x,
    cy: h.y,
    r: screenToImagePx(selected ? 5 : 4),
    fill: selected ? "#ffffff" : color,
    stroke: "#111111",
    "stroke-width": 1,
    "vector-effect": "non-scaling-stroke"
  }));
}

function updatePenSvg(){
  if(!penSvg){
    return;
  }
  penSvg.innerHTML = "";
  if(!baseImage.naturalWidth || !baseImage.naturalHeight){
    return;
  }
  penSvg.setAttribute("viewBox", `0 0 ${baseImage.naturalWidth} ${baseImage.naturalHeight}`);
  if(penPoints.length === 0){
    return;
  }

  const d = buildSvgPath(penPoints, isPenClosed);
  if(isPenClosed && penPoints.length >= 3){
    penSvg.appendChild(svgNode("path", {
      d,
      fill: mode === "fg" ? "rgba(0,200,120,0.18)" : "rgba(255,96,96,0.18)",
      stroke: "none"
    }));
  }

  penSvg.appendChild(svgNode("path", {
    d,
    fill: "none",
    stroke: isPenClosed ? "#14b8a6" : "#f59e0b",
    "stroke-width": 2,
    "stroke-dasharray": "5 4",
    "vector-effect": "non-scaling-stroke",
    "stroke-linecap": "round",
    "stroke-linejoin": "round"
  }));

  for(let i = 0; i < penPoints.length; i++){
    const p = penPoints[i];
    const selAnchor = dragObject && dragObject.type === "anchor" && dragObject.index === i;
    const selIn = dragObject && dragObject.type === "inHandle" && dragObject.index === i;
    const selOut = dragObject && dragObject.type === "outHandle" && dragObject.index === i;
    appendSvgHandle(p, "out", "#2dd4bf", selOut);
    appendSvgHandle(p, "in", "#fb7185", selIn);
    penSvg.appendChild(svgNode("circle", {
      cx: p.x,
      cy: p.y,
      r: screenToImagePx(selAnchor ? 5 : 4),
      fill: selAnchor ? "#00e676" : "#00aaff",
      stroke: "#111111",
      "stroke-width": 1,
      "vector-effect": "non-scaling-stroke"
    }));
  }

  if(!isPenClosed && penPoints.length >= 3){
    penSvg.appendChild(svgNode("circle", {
      cx: penPoints[0].x,
      cy: penPoints[0].y,
      r: screenToImagePx(9),
      fill: "none",
      stroke: "#111111",
      "stroke-width": 1,
      "stroke-dasharray": "2 2",
      "vector-effect": "non-scaling-stroke"
    }));
  }
}
function drawHandle(anchor, kind, color, selected){
  const h = kind === "out" ? anchor.outHandle : anchor.inHandle;
  if(isPoint(h, anchor.x, anchor.y, 0.75)){
    return;
  }
  overlayCtx.strokeStyle = color;
  overlayCtx.lineWidth = screenToImagePx(1);
  overlayCtx.setLineDash([]);
  overlayCtx.beginPath();
  overlayCtx.moveTo(anchor.x, anchor.y);
  overlayCtx.lineTo(h.x, h.y);
  overlayCtx.stroke();

  overlayCtx.fillStyle = selected ? "#ffffff" : color;
  overlayCtx.beginPath();
  overlayCtx.arc(h.x, h.y, screenToImagePx(selected ? 5 : 4), 0, Math.PI * 2);
  overlayCtx.fill();
  overlayCtx.strokeStyle = "#111111";
  overlayCtx.stroke();
}

function drawPenPath(withBase=true){
  if(withBase){
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    redrawOverlayBase();
  }
  updatePenSvg();
  updateToolButtons();
}
function clearPenPath(silent){
  if(!silent && penPoints.length > 0){
    pushUndo();
  }
  penPoints = [];
  isPenClosed = false;
  dragObject = null;
  lastPointer = null;
  if(!silent){
    redrawOverlay();
    document.getElementById("status").textContent = "Pen path cleared. Ctrl+Z restores it.";
  }
  updateToolButtons();
}

function clearMaskCanvas(){
  pushUndo();
  maskCtx.globalCompositeOperation = "source-over";
  maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
  maskDirty = true;
  penPoints = [];
  isPenClosed = false;
  dragObject = null;
  selectedPenObject = null;
  redrawOverlay();
  document.getElementById("status").textContent = "Mask canvas cleared. Ctrl+Z restores it.";
}

function canvasHitTestPenObject(x, y){
  const handleHit = screenToImagePx(handleRadius);
  const anchorHit = screenToImagePx(hitRadius);
  for(let i = penPoints.length - 1; i >= 0; i--){
    const p = penPoints[i];
    if(!handlesCollapsed(p)){
      if(isPoint(p.outHandle, x, y, handleHit)){
        return { type: "outHandle", index: i };
      }
      if(isPoint(p.inHandle, x, y, handleHit)){
        return { type: "inHandle", index: i };
      }
    }
  }
  for(let i = penPoints.length - 1; i >= 0; i--){
    if(isPoint(penPoints[i], x, y, anchorHit)){
      return { type: "anchor", index: i };
    }
  }
  return null;
}

function distanceToLineSegment(p, a, b){
  const vx = b.x - a.x;
  const vy = b.y - a.y;
  const wx = p.x - a.x;
  const wy = p.y - a.y;
  const len2 = vx * vx + vy * vy;
  let t = len2 > 0 ? (wx * vx + wy * vy) / len2 : 0;
  t = Math.max(0, Math.min(1, t));
  const proj = { x: a.x + t * vx, y: a.y + t * vy };
  return Math.sqrt(dist2(p, proj));
}

function hitTestSegment(x, y){
  if(penPoints.length < 2){
    return null;
  }
  const p = { x, y };
  const segCount = isPenClosed ? penPoints.length : penPoints.length - 1;
  let best = null;
  for(let i = 0; i < segCount; i++){
    const seg = getSegment(i);
    if(!seg){
      continue;
    }
    let prev = seg.p0;
    for(let step = 1; step <= sampleSteps; step++){
      const t = step / sampleSteps;
      const cur = cubicPoint(seg.p0, seg.c1, seg.c2, seg.p3, t);
      const d = distanceToLineSegment(p, prev, cur);
      if(!best || d < best.distance){
        best = { segment: i, t: (step - 0.5) / sampleSteps, distance: d };
      }
      prev = cur;
    }
  }
  if(best && best.distance <= screenToImagePx(segmentHitRadius)){
    return best;
  }
  return null;
}

function splitCubic(segmentIndex, t){
  const n = penPoints.length;
  const i = segmentIndex;
  const nextIdx = (i + 1) % n;
  const p0 = penPoints[i];
  const p3 = penPoints[nextIdx];
  const p1 = p0.outHandle;
  const p2 = p3.inHandle;

  const lerp = (a, b) => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });
  const p01 = lerp(p0, p1);
  const p12 = lerp(p1, p2);
  const p23 = lerp(p2, p3);
  const p012 = lerp(p01, p12);
  const p123 = lerp(p12, p23);
  const mid = lerp(p012, p123);

  p0.outHandle = p01;
  p3.inHandle = p23;
  const inserted = {
    x: mid.x,
    y: mid.y,
    inHandle: p012,
    outHandle: p123,
    smooth: true
  };

  if(isPenClosed && i === n - 1){
    penPoints.push(inserted);
  } else {
    penPoints.splice(i + 1, 0, inserted);
  }
  return inserted;
}

function hidePenContextMenu(){
  contextAnchorIndex = -1;
  penContextMenu.style.display = "none";
}

function showPenContextMenu(clientX, clientY, anchorIndex){
  contextAnchorIndex = anchorIndex;
  if(penPoints[anchorIndex]){
    penCurveToggleBtn.textContent = handlesCollapsed(penPoints[anchorIndex]) ? "Convert to Curve" : "Convert to Straight";
  }
  penContextMenu.style.left = clientX + "px";
  penContextMenu.style.top = clientY + "px";
  penContextMenu.style.display = "block";
}

function deleteContextAnchor(){
  if(contextAnchorIndex >= 0){
    pushUndo();
    deleteAnchor(contextAnchorIndex);
  }
  hidePenContextMenu();
}


function convertAnchorToCurve(index){
  if(index < 0 || index >= penPoints.length || penPoints.length < 2){
    return;
  }
  const anchor = penPoints[index];
  const n = penPoints.length;
  const prev = index > 0 ? penPoints[index - 1] : (isPenClosed ? penPoints[n - 1] : null);
  const next = index < n - 1 ? penPoints[index + 1] : (isPenClosed ? penPoints[0] : null);

  let tx = 0;
  let ty = 0;
  let handleLen = 32;
  if(prev && next){
    tx = next.x - prev.x;
    ty = next.y - prev.y;
    handleLen = Math.min(Math.sqrt(dist2(anchor, prev)), Math.sqrt(dist2(anchor, next))) / 3;
  } else if(next){
    tx = next.x - anchor.x;
    ty = next.y - anchor.y;
    handleLen = Math.sqrt(dist2(anchor, next)) / 3;
  } else if(prev){
    tx = anchor.x - prev.x;
    ty = anchor.y - prev.y;
    handleLen = Math.sqrt(dist2(anchor, prev)) / 3;
  }

  const len = Math.sqrt(tx * tx + ty * ty) || 1;
  const ux = tx / len;
  const uy = ty / len;
  handleLen = Math.max(12, Math.min(80, handleLen));
  anchor.inHandle = { x: anchor.x - ux * handleLen, y: anchor.y - uy * handleLen };
  anchor.outHandle = { x: anchor.x + ux * handleLen, y: anchor.y + uy * handleLen };
  anchor.smooth = true;
  selectedPenObject = { type: "anchor", index };
  drawPenPath();
  document.getElementById("status").textContent = "Anchor converted to curve. Drag handles to refine.";
}

function convertAnchorToStraight(index){
  if(index < 0 || index >= penPoints.length){
    return;
  }
  const anchor = penPoints[index];
  anchor.inHandle = { x: anchor.x, y: anchor.y };
  anchor.outHandle = { x: anchor.x, y: anchor.y };
  anchor.smooth = false;
  selectedPenObject = { type: "anchor", index };
  drawPenPath();
  document.getElementById("status").textContent = "Anchor converted to straight. Right-click again to make it curved.";
}

function toggleContextAnchorCurve(){
  if(contextAnchorIndex >= 0 && penPoints[contextAnchorIndex]){
    pushUndo();
    if(handlesCollapsed(penPoints[contextAnchorIndex])){
      convertAnchorToCurve(contextAnchorIndex);
    } else {
      convertAnchorToStraight(contextAnchorIndex);
    }
  }
  hidePenContextMenu();
}
function deleteAnchor(index){
  if(index < 0 || index >= penPoints.length){
    return;
  }
  if(penPoints.length <= 1){
    penPoints = [];
    isPenClosed = false;
  } else {
    penPoints.splice(index, 1);
    if(penPoints.length < 3){
      isPenClosed = false;
    }
  }
  dragObject = null;
  selectedPenObject = null;
  drawPenPath();
  document.getElementById("status").textContent = "Anchor deleted. Ctrl+Z restores it.";
}

function closePenPath(){
  if(penPoints.length < 3){
    document.getElementById("status").textContent = "Need at least 3 points to close path.";
    return;
  }
  if(!isPenClosed){
    pushUndo();
  }
  isPenClosed = true;
  dragObject = null;
  selectedPenObject = null;
  drawPenPath();
  document.getElementById("status").textContent = "Path closed. Click Apply Pen or press Enter to merge with the mask.";
}

function applyPenToMask(){
  if(!isPenClosed || penPoints.length < 3){
    document.getElementById("status").textContent = "Need a closed path before applying.";
    return;
  }
  pushUndo();
  const w = maskCanvas.width;
  const h = maskCanvas.height;
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = w;
  tempCanvas.height = h;
  const tctx = tempCanvas.getContext("2d");
  tctx.fillStyle = "#fff";
  tctx.beginPath();
  buildBezierPath(tctx, penPoints, true);
  tctx.closePath();
  tctx.fill();

  const pathData = tctx.getImageData(0, 0, w, h).data;
  const mm = maskCtx.getImageData(0, 0, w, h);
  const d = mm.data;
  for(let i = 3; i < pathData.length; i += 4){
    if(pathData[i] > 8){
      const j = i - 3;
      const v = mode === "fg" ? 255 : 0;
      d[j] = v;
      d[j + 1] = v;
      d[j + 2] = v;
      d[j + 3] = v;
    }
  }
  maskCtx.putImageData(mm, 0, 0);
  maskDirty = true;
  penPoints = [];
  isPenClosed = false;
  dragObject = null;
  selectedPenObject = null;
  redrawOverlay();
  document.getElementById("status").textContent = mode === "fg" ? "Pen selection merged into mask." : "Pen selection removed from mask.";
}

function beginNewAnchor(p){
  const anchor = createAnchor(p.x, p.y);
  penPoints.push(anchor);
  isPenClosed = false;
  dragObject = {
    type: "newAnchor",
    index: penPoints.length - 1,
    start: p,
    moved: false,
    undoPushed: true
  };
  lastPointer = p;
}

function onPointerDown(e){
  const resizeHandle = getResizeHandle(e);
  if(e.button === 0 && resizeHandle){
    beginImageResize(e, resizeHandle);
    return;
  }
  const p = getPos(e);
  if(e.button === 0){
    hidePenContextMenu();
  }
  if(tool === "brush"){
    if(!brushDirty){
      pushUndo();
      brushDirty = true;
    }
    isMouseDown = true;
    applyBrush(p.x, p.y);
    return;
  }

  if(e.button !== 0){
    return;
  }
  e.preventDefault();

  const hit = canvasHitTestPenObject(p.x, p.y);
  if(hit && hit.type === "anchor" && hit.index === 0 && !isPenClosed && penPoints.length >= 3){
    closePenPath();
    return;
  }

  if(hit){
    pushUndo();
    const anchor = penPoints[hit.index];
    selectedPenObject = { ...hit };
    dragObject = {
      ...hit,
      start: p,
      last: p,
      moved: false,
      symmetric: anchor.smooth && !e.altKey,
      undoPushed: true
    };
    if(hit.type === "inHandle" || hit.type === "outHandle"){
      anchor.smooth = !e.altKey;
    }
    isMouseDown = true;
    overlayCanvas.style.cursor = "move";
    drawPenPath();
    return;
  }

  const segHit = hitTestSegment(p.x, p.y);
  if(segHit){
    pushUndo();
    const inserted = splitCubic(segHit.segment, segHit.t);
    dragObject = {
      type: "anchor",
      index: penPoints.indexOf(inserted),
      start: p,
      last: p,
      moved: false,
      suppressClickDelete: true,
      undoPushed: true
    };
    selectedPenObject = { type: "anchor", index: penPoints.indexOf(inserted) };
    isMouseDown = true;
    overlayCanvas.style.cursor = "move";
    drawPenPath();
    document.getElementById("status").textContent = "Anchor added on path.";
    return;
  }

  pushUndo();
  beginNewAnchor(p);
  selectedPenObject = { type: "anchor", index: penPoints.length - 1 };
  isMouseDown = true;
  drawPenPath();
}

function updatePenCursor(e){
  if(tool !== "pen" || isMouseDown){
    return;
  }
  if(updateResizeCursor(e)){
    return;
  }
  const p = getPos(e);
  const hit = canvasHitTestPenObject(p.x, p.y);
  if(hit){
    overlayCanvas.style.cursor = "move";
    return;
  }
  if(hitTestSegment(p.x, p.y)){
    overlayCanvas.style.cursor = "copy";
    return;
  }
  overlayCanvas.style.cursor = "crosshair";
}

function onResizeWindowMove(e){
  if(resizeDrag && e.target !== overlayCanvas){
    applyImageResize(e);
  }
}

function onResizeWindowUp(e){
  if(resizeDrag){
    onPointerUp(e);
  }
}

function onPointerMove(e){
  if(resizeDrag){
    applyImageResize(e);
    return;
  }
  if(updateResizeCursor(e)){
    return;
  }
  const p = getPos(e);
  if(tool === "brush"){
    brushPreview = p;
    if(!isMouseDown){
      redrawOverlay();
      return;
    }
  } else if(!isMouseDown){
    updatePenCursor(e);
    return;
  }
  if(tool === "brush"){
    applyBrush(p.x, p.y);
    return;
  }
  if(!dragObject || dragObject.index < 0 || dragObject.index >= penPoints.length){
    return;
  }

  const idx = dragObject.index;
  const anchor = penPoints[idx];
  const movedEnough = Math.sqrt(dist2(p, dragObject.start || p)) > dragThreshold;
  dragObject.moved = dragObject.moved || movedEnough;

  if(dragObject.type === "newAnchor"){
    if(movedEnough){
      anchor.smooth = true;
      anchor.outHandle.x = p.x;
      anchor.outHandle.y = p.y;
      reflectHandle(anchor, anchor.outHandle, anchor.inHandle);
    }
  } else if(dragObject.type === "anchor"){
    const last = dragObject.last || p;
    const dx = p.x - last.x;
    const dy = p.y - last.y;
    anchor.x += dx;
    anchor.y += dy;
    anchor.inHandle.x += dx;
    anchor.inHandle.y += dy;
    anchor.outHandle.x += dx;
    anchor.outHandle.y += dy;
    dragObject.last = p;
  } else if(dragObject.type === "inHandle"){
    anchor.inHandle.x = p.x;
    anchor.inHandle.y = p.y;
    anchor.smooth = !e.altKey;
    if(anchor.smooth){
      reflectHandle(anchor, anchor.inHandle, anchor.outHandle);
    }
  } else if(dragObject.type === "outHandle"){
    anchor.outHandle.x = p.x;
    anchor.outHandle.y = p.y;
    anchor.smooth = !e.altKey;
    if(anchor.smooth){
      reflectHandle(anchor, anchor.outHandle, anchor.inHandle);
    }
  }
  drawPenPath();
}

function onPointerUp(e){
  if(resizeDrag){
    window.removeEventListener("mousemove", onResizeWindowMove);
    window.removeEventListener("mouseup", onResizeWindowUp);
    flushResizeUpdate();
    resizeDrag = null;
    isMouseDown = false;
    redrawOverlay();
    updateResizeCursor(e) || updateToolButtons();
    return;
  }
  if(tool === "pen" && dragObject && dragObject.type === "newAnchor" && !dragObject.moved){
    const idx = dragObject.index;
    if(penPoints[idx]){
      penPoints[idx].smooth = false;
      penPoints[idx].inHandle = { x: penPoints[idx].x, y: penPoints[idx].y };
      penPoints[idx].outHandle = { x: penPoints[idx].x, y: penPoints[idx].y };
    }
  }
  isMouseDown = false;
  brushDirty = false;
  dragObject = null;
  lastPointer = null;
  overlayCtx.globalCompositeOperation = "source-over";
  maskCtx.globalCompositeOperation = "source-over";
  if(tool === "pen"){
    drawPenPath();
    updatePenCursor(e);
  }
}

overlayCanvas.addEventListener("mousedown", onPointerDown);
overlayCanvas.addEventListener("mousemove", onPointerMove);
overlayCanvas.addEventListener("mouseenter", (e) => {
  if(tool === "brush"){
    brushPreview = getPos(e);
    redrawOverlay();
  }
});
overlayCanvas.addEventListener("mouseleave", () => {
  if(tool === "brush"){
    brushPreview = null;
    redrawOverlay();
  }
});
brushInput.addEventListener("input", () => {
  if(tool === "brush"){
    redrawOverlay();
  }
});
uploadImagesInput.addEventListener("change", uploadImages);
overlayCanvas.addEventListener("contextmenu", (e) => {
  if(tool !== "pen"){
    return;
  }
  e.preventDefault();
  const p = getPos(e);
  const hit = canvasHitTestPenObject(p.x, p.y);
  if(hit){
    selectedPenObject = { type: hit.type, index: hit.index };
    showPenContextMenu(e.clientX, e.clientY, hit.index);
    drawPenPath();
  } else {
    hidePenContextMenu();
  }
});
window.addEventListener("mouseup", onPointerUp);
window.addEventListener("click", (e) => {
  if(!penContextMenu.contains(e.target)){
    hidePenContextMenu();
  }
});

overlayCanvas.addEventListener("dblclick", () => {
  if(tool !== "pen") return;
  applyPenToMask();
});

window.addEventListener("keydown", (e) => {
  const activeTag = (document.activeElement && document.activeElement.tagName || "").toLowerCase();
  if(activeTag === "input" && !(e.ctrlKey || e.metaKey)){
    return;
  }

  if((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key.toLowerCase() === "z"){
    undoEdit();
    e.preventDefault();
    return;
  }
  if((e.ctrlKey || e.metaKey) && (e.key.toLowerCase() === "k" || e.key === "Backspace" || e.key === "Delete")){
    clearMaskCanvas();
    e.preventDefault();
    return;
  }

  if(tool !== "pen"){
    return;
  }
  if(e.key === "Escape"){
    clearPenPath();
    e.preventDefault();
  } else if(e.key === "Enter"){
    applyPenToMask();
    e.preventDefault();
  } else if(e.key.toLowerCase() === "c"){
    closePenPath();
    e.preventDefault();
  } else if((e.key === "Backspace" || e.key === "Delete") && selectedPenObject){
    pushUndo();
    deleteAnchor(selectedPenObject.index);
    e.preventDefault();
  }
});

async function uploadImages(){
  const files = Array.from(uploadImagesInput.files || []);
  if(files.length === 0){
    return;
  }
  await syncCurrentMask();
  const form = new FormData();
  for(const file of files){
    form.append("images", file);
  }
  const r = await fetch("/api/upload", { method: "POST", body: form }).then(x=>x.json());
  uploadImagesInput.value = "";
  if(r.ok && r.added > 0){
    totalFrames = r.total_frames;
    document.getElementById("frameCount").textContent = String(totalFrames);
    await loadFrame(r.first_index, true);
  } else {
    document.getElementById("status").textContent = "upload failed: " + (r.msg || "no valid images");
  }
}

async function syncCurrentMask(){
  if(!maskDirty || maskCanvas.width === 0 || maskCanvas.height === 0){
    return;
  }
  const maskData = maskCanvas.toDataURL("image/png");
  const r = await fetch("/api/update_mask", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ index: currentFrame, mask: maskData })
  }).then(x=>x.json());
  if(r.ok){
    maskDirty = false;
  } else {
    document.getElementById("status").textContent = "sync failed: " + (r.msg || "unknown");
  }
}

async function saveFrame(){
  const status = document.getElementById("status");
  status.textContent = "saving...";
  const maskData = maskCanvas.toDataURL("image/png");
  const r = await fetch("/api/save", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ index: currentFrame, mask: maskData })
  }).then(x=>x.json());
  if(r.ok){
    maskDirty = false;
    status.textContent = "saved: " + r.overlay + ", " + r.cutout;
  } else {
    status.textContent = "save failed: " + (r.msg || "unknown");
  }
}

async function resetFrame(){
  pushUndo();
  const r = await fetch(`/api/reset/${currentFrame}`, { method:"POST" }).then(x=>x.json());
  if(r.ok){
    renderOverlayFromBaseMask(r.mask);
    maskDirty = false;
    clearPenPath(true);
  }
}

async function prevFrame(){ if(currentFrame > 0) await loadFrame(currentFrame - 1); }
async function nextFrame(){ if(currentFrame + 1 < totalFrames) await loadFrame(currentFrame + 1); }

function updateSize(){
  if(!baseImage || !baseImage.naturalWidth || !baseImage.naturalHeight){
    return;
  }
  const wrap = document.querySelector(".canvas-wrap");
  const rect = wrap.getBoundingClientRect();
  const pad = 24;
  const availableW = Math.max(120, rect.width - pad);
  const availableH = Math.max(120, rect.height - pad);
  imageFitScale = Math.min(availableW / baseImage.naturalWidth, availableH / baseImage.naturalHeight, 1);
  const scale = Math.max(0.05, imageFitScale * imageScaleFactor);
  const displayW = Math.max(1, Math.round(baseImage.naturalWidth * scale));
  const displayH = Math.max(1, Math.round(baseImage.naturalHeight * scale));
  baseImage.style.width = displayW + "px";
  baseImage.style.height = displayH + "px";
  overlayCanvas.style.width = displayW + "px";
  overlayCanvas.style.height = displayH + "px";
  penSvg.style.width = displayW + "px";
  penSvg.style.height = displayH + "px";
  penSvg.setAttribute("viewBox", `0 0 ${baseImage.naturalWidth} ${baseImage.naturalHeight}`);
  updatePenSvg();
}

window.addEventListener("resize", updateSize);

window.addEventListener("load", async () => {
  const first = await fetch("/api/list").then(r=>r.json());
  totalFrames = first.total_frames;
  document.getElementById("frameCount").textContent = String(totalFrames);
  if(totalFrames > 0){
    await loadFrame(0, true);
  }
  updateModeButtons();
  updateToolButtons();
  updateShortcutHelp();
  updateSize();
  document.getElementById("status").textContent = "Pen: 单击空白加直线锚点，按住拖动新锚点拉出贝塞尔手柄；左键拖锚点/手柄移动或调弧度；点路径增点；右键锚点/手柄可在直线/曲线间切换或删除锚点；点起点闭合，Enter 生成并合并选区。";
});
</script>
</body>
</html>
"""

    @app.route("/api/list")
    def api_list():
        return jsonify({"total_frames": len(frame_data), "frames": [f["image_name"] for f in frame_data]})

    @app.route("/api/upload", methods=["POST"])
    def api_upload():
        files = request.files.getlist("images")
        if not files:
            return jsonify({"ok": False, "msg": "no files"}), 400

        upload_dir = output_root / args.category / "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        batch_dir = upload_dir / f"batch_{len(frame_data):04d}_{int(time.time() * 1000)}"
        os.makedirs(batch_dir, exist_ok=True)
        first_index = len(frame_data)
        upload_items = []

        for file in files:
            raw = file.read()
            if not raw:
                continue
            arr = np.frombuffer(raw, dtype=np.uint8)
            img_any = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img_any is None:
                continue

            stem = Path(file.filename or f"upload_{len(frame_data):04d}").stem
            safe_stem = "".join(ch if ch.isalnum() or ch in ["-", "_"] else "_" for ch in stem).strip("_")
            if not safe_stem:
                safe_stem = f"upload_{len(frame_data):04d}"
            image_name = f"{len(upload_items):05d}_{safe_stem}.png"
            image_path = batch_dir / image_name

            if img_any.ndim == 2:
                image = cv2.cvtColor(img_any, cv2.COLOR_GRAY2BGR)
            elif img_any.shape[2] == 4:
                image = cv2.cvtColor(img_any, cv2.COLOR_BGRA2BGR)
            else:
                image = img_any[:, :, :3]

            cv2.imwrite(str(image_path), image)
            upload_items.append({"image_path": image_path, "image_name": image_name, "image": image})

        if len(upload_items) == 0:
            return jsonify({"ok": False, "msg": "no valid images"}), 400

        try:
            masks = infer_masks_for_images(batch_dir, [item["image"] for item in upload_items])
        except Exception as exc:
            return jsonify({"ok": False, "msg": f"inference failed: {exc}"}), 500

        for item, mask in zip(upload_items, masks):
            frame_data.append(
                {
                    "image_path": str(item["image_path"]),
                    "image_name": item["image_name"],
                    "image": item["image"],
                    "base_mask": mask.copy(),
                    "mask": mask.copy(),
                }
            )

        return jsonify({"ok": True, "added": len(upload_items), "first_index": first_index, "total_frames": len(frame_data)})

    @app.route("/api/frame/<int:idx>")
    def api_frame(idx):
        if idx < 0 or idx >= len(frame_data):
            return jsonify({"ok": False, "msg": "invalid frame"}), 400
        item = frame_data[idx]
        image_b64 = get_data_url(item["image"], "jpg")
        # return current editable mask for this frame as grayscale-as-rgba PNG
        mask = item["mask"]
        mask_rgba = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
        mask_rgba[:, :, 3] = mask
        mask_b64 = get_data_url(mask_rgba, include_alpha=True)
        return jsonify({
            "ok": True,
            "index": idx,
            "total_frames": len(frame_data),
            "image": image_b64,
            "mask": mask_b64
        })

    @app.route("/api/reset/<int:idx>", methods=["POST"])
    def api_reset(idx):
        if idx < 0 or idx >= len(frame_data):
            return jsonify({"ok": False, "msg": "invalid frame"}), 400
        item = frame_data[idx]
        item["mask"] = item["base_mask"].copy()
        mask = item["mask"]
        mask_rgba = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
        mask_rgba[:, :, 3] = mask
        return jsonify({"ok": True, "mask": get_data_url(mask_rgba, include_alpha=True)})

    def _decode_mask_data_url(mask_data_url):
        if "," not in mask_data_url:
            raise ValueError("invalid mask payload")
        _, payload = mask_data_url.split(",", 1)
        raw = np.frombuffer(base64.b64decode(payload), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("failed to decode mask")
        if img.ndim == 3 and img.shape[2] >= 3:
            # The editable mask stores selected pixels as white/red channel values; alpha can be nonzero for erased black pixels.
            m = img[:, :, 0]
        else:
            m = img
        m = (m > 127).astype(np.uint8) * 255
        return m

    @app.route("/api/update_mask", methods=["POST"])
    def api_update_mask():
        payload = request.get_json(force=True)
        idx = int(payload.get("index", -1))
        mask_data = payload.get("mask", "")
        if idx < 0 or idx >= len(frame_data):
            return jsonify({"ok": False, "msg": "invalid frame"}), 400
        try:
            edited_mask = _decode_mask_data_url(mask_data)
        except Exception as exc:
            return jsonify({"ok": False, "msg": str(exc)}), 400
        item = frame_data[idx]
        img = item["image"]
        edited_mask = cv2.resize(edited_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        item["mask"] = edited_mask
        return jsonify({"ok": True})

    @app.route("/api/save", methods=["POST"])
    def api_save():
        payload = request.get_json(force=True)
        idx = int(payload.get("index", -1))
        mask_data = payload.get("mask", "")
        if idx < 0 or idx >= len(frame_data):
            return jsonify({"ok": False, "msg": "invalid frame"}), 400
        try:
            edited_mask = _decode_mask_data_url(mask_data)
        except Exception as exc:
            return jsonify({"ok": False, "msg": str(exc)}), 400

        item = frame_data[idx]
        img = item["image"]
        edited_mask = cv2.resize(edited_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        item["mask"] = edited_mask

        cutout = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
        cutout[:, :, :3] = img[:, :, :3]
        cutout[:, :, 3] = edited_mask
        overlay = overlay_mask_blue(img, edited_mask)

        name = Path(item["image_path"]).stem
        cut_name = f"{name}.{args.output_format.lower()}"
        overlay_name = f"{name}.{args.overlay_format.lower()}"
        cut_path = os.path.join(str(cutout_dir), cut_name)
        overlay_path = os.path.join(str(overlay_dir), overlay_name)

        if args.output_format.lower() == "png":
            cv2.imwrite(cut_path, cutout)
        else:
            alpha_f = cutout[:, :, 3:4] / 255.0
            bg = np.zeros_like(img)
            rgb = (cutout[:, :, :3] * alpha_f + bg * (1 - alpha_f)).astype(np.uint8)
            cv2.imwrite(cut_path, rgb)

        cv2.imwrite(overlay_path, overlay)
        return jsonify({"ok": True, "overlay": Path(overlay_path).name, "cutout": Path(cut_path).name})

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

















