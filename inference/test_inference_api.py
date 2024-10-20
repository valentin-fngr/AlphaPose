from inference_api import Skeleton2DInference

config = "configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml"
checkpoint = "pretrained_models/halpe26_fast_res50_256x192.pth"
video_path = "example_videos/dance_3_people.mp4"


skeleton_2d_detector = Skeleton2DInference(
    config, 
    checkpoint, 
    device="cuda", 
    debug=True
)

results = skeleton_2d_detector.inference(video_path)