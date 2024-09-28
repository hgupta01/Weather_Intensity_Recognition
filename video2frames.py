import os
import cv2
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm

def vid2frames(vid_file, save_folder, prefix="{:03d}.jpg"):
    vidcap = cv2.VideoCapture(vid_file)
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for idx in range(num_frames):
        ret, image = vidcap.read()
        if ret:
            cv2.imwrite(os.path.join(save_folder, prefix.format(idx)), image)     # save frame as JPEG file
            

if __name__ == '__main__':
    video_files = sorted(Path("dataset/videos/videos").glob("*.mp4"))
    save_top_folder = Path("dataset/videos/frames/")

    if not save_top_folder.exists():
        os.mkdir(str(save_top_folder))

    for video_file in tqdm(video_files, desc="Number of videos"):
        save_vid_folder = save_top_folder / video_file.stem
        if not save_vid_folder.exists():
            os.mkdir(save_vid_folder)

        if not any(save_vid_folder.iterdir()):
            vid2frames(str(video_file), str(save_vid_folder))