import os
import cv2
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm

cv2.setNumThreads(8)

def flow_to_img(raw_flow, bound=20.):
    """Convert flow to gray image.

    Args:
        raw_flow (np.ndarray[float]): Estimated flow with the shape (w, h).
        bound (float): Bound for the flow-to-image normalization. Default: 20.

    Returns:
        np.ndarray[uint8]: The result list of np.ndarray[uint8], with shape
                        (w, h).
    """
    flow = np.clip(raw_flow, -bound, bound)
    flow += bound
    flow *= (255 / float(2 * bound))
    flow = flow.astype(np.uint8)
    return flow


def generate_flow(frames, method='tvl1'):
    """Estimate flow with given frames.

    Args:
        frames (list[np.ndarray[uint8]]): List of rgb frames, with shape
                                        (w, h, 3).
        method (str): Use which method to generate flow. Options are 'tvl1'
                    and 'farneback'. Default: 'tvl1'.

    Returns:
        list[np.ndarray[float]]: The result list of np.ndarray[float], with
                                shape (w, h, 2).
    """
    assert method in ['tvl1', 'farneback', 's2d', 'deepflow']
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

    if method == 'tvl1':
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        def op(x, y):
            return tvl1.calc(x, y, None)

    elif method == 's2d':
        def op(x, y):
            return cv2.optflow.calcOpticalFlowSparseToDense(x, 
                                                            y, 
                                                            None, 
                                                            grid_step = 6,
                                                            k = 128,
                                                            sigma = 0.05,
                                                            use_post_proc = True,
                                                            fgs_lambda = 500.0,
                                                            fgs_sigma = 1.5)

    elif method == 'farneback':
        def op(x, y):
            return cv2.calcOpticalFlowFarneback(x, y, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    elif method == 'deepflow':
        deepflow = cv2.optflow.createOptFlow_DeepFlow()
        def op(x,y):
            return deepflow.calc(x, y, None)
            
    gray_st = gray_frames[:-1]
    gray_ed = gray_frames[1:]

    flow = [op(x, y) for x, y in zip(gray_st, gray_ed)]
    return flow


def extract_dense_flow(path,
                       dest,
                       bound=20.,
                       save_rgb=False,
                       start_idx=0,
                       flow_tmpl='{}_{:03d}.jpg',
                       method='tvl1',
                       do_ms=False):
    """Extract dense flow given video or frames, save them as gray-scale
    images.

    Args:
        path (str): Location of the input video.
        dest (str): The directory to store the extracted flow images.
        bound (float): Bound for the flow-to-image normalization. Default: 20.
        save_rgb (bool): Save extracted RGB frames. Default: False.
        start_idx (int): The starting frame index if use frames as input, the
            first image is path.format(start_idx). Default: 0.
        rgb_tmpl (str): The template of RGB frame names, Default:
            'img_{:05d}.jpg'.
        flow_tmpl (str): The template of Flow frame names, Default:
            '{}_{:05d}.jpg'.
        method (str): Use which method to generate flow. Options are 'tvl1'
            and 'farneback'. Default: 'tvl1'.
        do_ms (bool): perform mean substraction
    """
    assert path.exists()
    if not dest.exists():
        dest.mkdir()
        
    video = cv2.VideoCapture(str(path))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    for _ in range(num_frames):
        ret, image = video.read()
        if ret:
            frames.append(image)
        
    flow = generate_flow(frames, method=method)
    if do_ms:
        mean_flow = np.stack(flow).mean(axis=0, keepdims=True)
        flow -= mean_flow

    flow_x = [flow_to_img(x[:, :, 0], bound) for x in flow]
    flow_y = [flow_to_img(x[:, :, 1], bound) for x in flow]

    if not os.path.exists(dest):
        os.system('mkdir -p ' + dest)
        
    flow_x_names = [
        os.path.join(dest, flow_tmpl.format('x', ind + start_idx))
        for ind in range(len(flow_x))
    ]
    flow_y_names = [
        os.path.join(dest, flow_tmpl.format('y', ind + start_idx))
        for ind in range(len(flow_y))
    ]

    num_frames = len(flow)
    for i in range(num_frames):
        cv2.imwrite(flow_x_names[i], flow_x[i])
        cv2.imwrite(flow_y_names[i], flow_y[i])


if __name__ == '__main__':
    video_files = sorted(Path("dataset/videos/videos").glob("*.mp4"))
    save_flow_folder = Path("dataset/videos/flows/")
    if not save_flow_folder.exists():
        os.mkdir(str(save_flow_folder))
    
    bound = 1
    start_idx = 1
    for video_file in tqdm(video_files, desc="Number of videos"):
        save_vid_folder = save_flow_folder / video_file.stem
        if not save_vid_folder.exists():
            try:
                extract_dense_flow(video_file, save_vid_folder, bound, start_idx, method='s2d', do_ms=True)
            except:
                print(f'some problem {str(save_vid_folder)}. Trying tvl1')
                extract_dense_flow(video_file, save_vid_folder, bound, start_idx, method='tvl1', do_ms=True)