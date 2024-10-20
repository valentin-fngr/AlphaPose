import sys
import gc
import os
import argparse
from tqdm import tqdm 
import time

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from omegaconf import DictConfig

import torch
from detector.apis import get_detector
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.config import update_config
from alphapose.models import builder
from alphapose.utils.writer import DEFAULT_VIDEO_SAVE_OPT as video_save_opt
from alphapose.utils.writer import DataWriter
from alphapose.utils.vis import getTime
from alphapose.utils.transforms import flip, flip_heatmap


class Skeleton2DInference: 

    def __init__(self, config, checkpoint, device, qsize=10, debug=False): 
        self.cfg = update_config(config)
        self.device = device
        self.detbatch = 1 
        self.posebatch = 1 
        self.qsize = qsize
        self.detector = "yolox-l"

        # NOTE : due to the general design of alphapose, we have to recreate args here. 
        # This is very redundant as it carries information already available in the constructor. 
        # But this is the simplest way for now ...

        args = DictConfig(dict(
            outputpath="debug" if debug else None,
            sp=True,
            detector=self.detector,
            save_img=True if debug else False,
            vis=False,
            showbox=False,
            profile=False,
            format=None,
            min_box_area=0,
            detbatch=1,
            posebatch=1,
            eval=False,
            gpus=[0],
            qsize=self.qsize,
            flip=False,
            debug=False,
            webcam=-1,
            save_video=False,
            vis_fast=False,
            pose_flow=False,
            pose_track=True, 
            device=device
        ))


        if args.pose_track:
            self.tracker = Tracker(tcfg, args)

        self.args = argparse.Namespace(**args)
        
        # TODO : this is terrible, we should do this before !
        self.args.gpus = args.gpus if torch.cuda.device_count() >= 1 else [-1]
        print(self.args.gpus)
        self.args.tracking = True
        self.args.cfg = self.cfg



        self.pose_model = builder.build_sppe(self.cfg.MODEL, preset_cfg=self.cfg.DATA_PRESET)
        self.pose_model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        self.pose_model.to(device)
        self.pose_model.eval()
        
        self.pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
        self.writer = DataWriter(self.cfg, self.args, save_video=self.args.save_video, video_save_opt=video_save_opt, queueSize=self.qsize)


    def _check_input(self, video_path):         
        if os.path.isfile(video_path):
            videofile = video_path
            return 'video', videofile
        else:
            raise IOError('Error: --video must refer to a video file, not directory.')

    def print_finish_info(self, args):
        print('===========================> Finish Model Running.')
        if (args.save_img or args.save_video) and not args.vis_fast:
            print('===========================> Rendering remaining images in the queue...')
            print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')

    def inference(self, video_path): 
        """
        Performs 2d skeleton inference 

        Argument 
        ------ 
        video_path : str 
            The input video path 

        Output 
        ----- 
        final_results : List[Dict] 
            A dict containing 2D skeleton results. Each frame is formated as : 
                {
                    "image_id": "x.jpg", (useless)
                    "category_id" : x (useless), 
                    "keypoints": [x1, ...., xn], 
                    "score": x, 
                    "box": [x1, x2, x3, x4], 
                    "idx": i, 
                    "img_shpae" : [height, width]
                }
        """
        mode, input_source = self._check_input(video_path)

        # start writer 
        self.writer.start()

        det_loader = DetectionLoader(input_source, get_detector(self.args), self.cfg, self.args, batchSize=self.detbatch, mode=mode, queueSize=self.qsize)
        det_worker = det_loader.start()
        data_len = det_loader.length

        print('INFO : number of frames detected : ', len(range(data_len)))
        im_names_desc = tqdm(range(data_len), dynamic_ncols=True)
        batchSize = self.posebatch

        try:
            for i in im_names_desc:
                start_time = getTime()
                with torch.no_grad():
                    (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                    if orig_img is None:
                        break
                    if boxes is None or boxes.nelement() == 0:
                        self.writer.save(None, None, None, None, None, orig_img, im_name)
                        continue
                    # Pose Estimation
                    inps = inps.to(self.device)
                    datalen = inps.size(0)
                    leftover = 0
                    if (datalen) % batchSize:
                        leftover = 1
                    num_batches = datalen // batchSize + leftover
                    hm = []
                    for j in range(num_batches):
                        inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                        if self.args.flip:
                            inps_j = torch.cat((inps_j, self.flip(inps_j)))
                        hm_j = self.pose_model(inps_j)
                        if self.args.flip:
                            hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], self.pose_dataset.joint_pairs, shift=True)
                            hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                        hm.append(hm_j)
                    hm = torch.cat(hm)
                    if self.args.pose_track:
                        boxes,scores,ids,hm,cropped_boxes = track(self.tracker, self.args, orig_img, inps, boxes, hm, cropped_boxes, im_name, scores)

                    hm = hm.cpu()
                    self.writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
            self.print_finish_info(self.args)
            while(self.writer.running()):
                time.sleep(1)
                print('===========================> Rendering remaining ' + str(self.writer.count()) + ' images in the queue...', end='\r')
            final_results = self.writer.stop()
            det_loader.stop()
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(repr(e))
            print('An error as above occurs when processing the images, please check it')
            raise e
        print(f"Final results of {len(final_results)} frames")
        return final_results