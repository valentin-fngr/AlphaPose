"""Script for converting detector + pose model + tracker to onnx."""
import argparse
import os
import sys
import time



import numpy as np
from torch import nn
import torch.onnx
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (which should contain both the script and the detector module)
parent_dir = os.path.dirname(script_dir)

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

from detector.apis import get_detector
from trackers.tracker_cfg import cfg as tcfg
from trackers.tracker_api import Tracker
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.file_detector import FileDetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.writer import DataWriter

"""----------------------------- onnx convertion options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--cfg', type=str, required=True,
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='checkpoint file name')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolox-l")
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=False)

args = parser.parse_args()
cfg = update_config(args.cfg)

# one GPU is enough
args.gpus = [0] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")

batch_size = 4

# load detector (weights loaded)
detector = get_detector(args)
detector_inp_dim = detector.inp_dim
detector.load_model()
if isinstance(detector, torch.nn.DataParallel): 
    detector_model = detector.model.module
else: 
    detector_model = detector.model
    
# convert to onnx ,(b,3,h,w)
x = torch.randn((batch_size, 3, detector_inp_dim, detector_inp_dim), requires_grad=True, device=args.device)
out = detector_model(x)
torch.onnx.export(detector_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "detector.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=13,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})




# Load pose model (must load weights!)
pose_inp_w, pose_inp_h = cfg.DATA_PRESET['IMAGE_SIZE']
pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
print('Loading pose model from %s...' % (args.checkpoint,))
pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
x = torch.randn((batch_size, 3, pose_inp_w, pose_inp_h), requires_grad=True, device=args.device)
pose_model.to(args.device)
out = pose_model(x)
torch.onnx.export(pose_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "pose_model.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=13,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})


# Load pose track model (must load weights!)
if args.pose_track:
    tracker = Tracker(tcfg, args)
    tracker_model = tracker.model
    if isinstance(tracker_model, torch.nn.DataParallel): 
        tracker_model = tracker_model.module

tracker_model.to(args.device)

# use the same input shape as the pose model
out = tracker_model(x)

torch.onnx.export(tracker_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "tracker_model.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=13,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
