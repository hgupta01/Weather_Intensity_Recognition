import os
import cv2
from tqdm import tqdm
from pathlib import Path


def vid2frames(vid_file, save_folder, prefix="{:03d}.jpg"):
    vidcap = cv2.VideoCapture(vid_file)
    success, image = vidcap.read()
    count = 1
    success = True
    while success:
        cv2.imwrite(os.path.join(save_folder, prefix.format(count)),
                    image)     # save frame as JPEG file
        count += 1
        success, image = vidcap.read()


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
