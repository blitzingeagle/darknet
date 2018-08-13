import os
from glob import glob

def process_video(video, output_dir):
    # darknet detector demo cfg/coco.data cfg/yolov2.cfg weights/yolov2.weights /workspace/veryverysmall.avi -avg 1 -prefix results/frame
    cmd = "darknet detector demo cfg/coco.data cfg/yolov2.cfg weights/yolov2.weights \"{}\" -avg 1 -prefix \"{}\"".format(video, os.path.join(output_dir, "frame"))
    # cmd = "darknet detector demo cfg/license_plate.data cfg/license_plate.cfg weights/license_plate_final.weights {} -avg 1 -prefix {}".format(video, os.path.join(output_dir, "frame"))
    print(">", cmd)
    os.system(cmd)

def process_videos():
    input_dirs = sorted(glob("/workspace/input/*"))
    print(input_dirs)
    for input_dir in input_dirs:
        videos = sorted(glob(os.path.join(input_dir, "*.mp4")))

        for video in videos:
            output_dir = os.path.splitext(video)[0].replace("input", "frames")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if not os.path.isfile(os.path.join(output_dir, "frame.txt")):
                process_video(video, output_dir)

if __name__ == "__main__":
    process_videos()

