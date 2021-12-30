from os.path import join
import cv2
import numpy as np
from numpy import ndarray
from numpy.linalg import norm
from typing import List, Optional, Iterable

from mask import apply_mask

def write_video_from_frames(
    video_name: str = 'output',
    vid_format: str = 'mp4',
    fps: int = 120,
    frame_list: List[ndarray] = [], 
    mask: Optional[ndarray] = None):

    if vid_format == 'mp4':
        encode = 'mp4v'
    elif vid_format == 'avi':
        encode = 'XVID'
    else:
        raise NotImplementedError
    
    fourcc = cv2.VideoWriter_fourcc(*encode)

    name = '{}.{}'.format(video_name, vid_format)
    H, W, _ = frame_list[0].shape
    
    writer = cv2.VideoWriter(name, fourcc, fps, (W, H))
    for frame in frame_list:
        if mask is not None:
            out = apply_mask(frame, mask)
            writer.write(out)
        else:
            writer.write(frame)    
    writer.release()

def write_frames(
    frame_list: List[ndarray], 
    dir_path: str, 
    prefix: str = 'frame', 
    mask: List[ndarray] = None) -> None:
    
    for i, frame in enumerate(frame_list):
        path = join(dir_path, prefix + '_{}'.format(i))
        out = frame
        if mask is not None:
            out = apply_mask(frame, mask)
        
        cv2.imwrite(path, out)

def calculate_angle(
    pt1: ndarray, 
    pt2: ndarray, 
    pt3: ndarray, 
    use_degree: bool = True) -> float:
    vector_1 = pt1 - pt2
    vector_2 = pt3 - pt2
    
    angle = np.arccos(np.dot(vector_1, vector_2) / (norm(vector_1) * norm(vector_2)))
    if use_degree:
        angle = np.degrees(angle)
    return angle

def group_kps(keypoints):
    return [(keypoints[i], keypoints[i+1], keypoints[i+2]) for i in range(0, len(keypoints), 3)]


