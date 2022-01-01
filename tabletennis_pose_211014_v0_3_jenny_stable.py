import os
import sys
import cv2
import pdb
import time

import random
import argparse
import numpy as np
from ctypes import *
from queue import Queue
import pandas as pd

from threading import Thread, enumerate
from sklearn.metrics.pairwise import cosine_similarity

import copy
from darknet import darknet
from fun_2d_ne import *


from fore_back_hand import *
from backend_firebase import *
from visualization import *

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default=None,
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    parser.add_argument("--rotate", type=int, default=0,
                        help="rotate image")
    parser.add_argument("--bounce_mask", type=str, default=None,
                        help="bounce mask image, <bounce_mask>.jpg")
    parser.add_argument("--draw_original",  action='store_true',
                        help="draw on original frame")
    parser.add_argument("--fps_skip", type=int, default=1,
                        help="")
    parser.add_argument("--show_2d", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ref_pts", type=str, default=r'C:\Users\OWNER\Documents\Jenny\20210323\points_arr.npy',
                        help="table tennis 4 corner <>.npy")
    parser.add_argument("--bg_mask", type=str, default=None,
                        help="background mask image, <bounce_mask>.jpg")
    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(
            os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(
            os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(
            os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))



def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # 返回旋转后的图像
    return rotated


def video_capture(frame_queue, darknet_image_queue, frame_rgb_queue, pose_queue, fps_skip, time_queue, rotate_ang, bg_mask, frame_queue_toShowing):
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # if frame_count > start_frame:
        if (frame_count % fps_skip) == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if rotate_ang != 0:
                frame_rgb = rotate(frame_rgb, rotate_ang)  # rotate

            frame_resized = cv2.resize(frame_rgb, (width, height),
                                       interpolation=cv2.INTER_LINEAR)

            # frame_resized = cv2.putText(frame_resized, str(frame_count), (30, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
            #                     2, (0, 255, 255), 1, cv2.LINE_AA)
            frame_rgb_queue.put(frame_rgb)
            '''---'''

            frame_queue.put(frame_resized)
            pose_queue.put((frame_count, frame_resized))

            frame_queue_toShowing.put(copy.deepcopy(frame_rgb))

            frame_resized_mask = copy.deepcopy(frame_resized)

            if not bg_mask is None:
                bg_mask = cv2.resize(bg_mask, (width, height),
                                     interpolation=cv2.INTER_LINEAR)
                # import pdb
                # pdb.set_trace()
                frame_resized_mask = frame_resized_mask*bg_mask

            darknet.copy_image_from_bytes(
                darknet_image, frame_resized_mask.tobytes())
            darknet_image_queue.put(darknet_image)
            time_queue.put(time.time())
        frame_count += 1
    cap.release()


def inference(darknet_image_queue, detections_queue, fps_queue):
    while cap.isOpened():
        darknet_image = darknet_image_queue.get()
        prev_time = time.time()
        detections = darknet.detect_image(
            network, class_names, darknet_image, thresh=args.thresh)
        # for d in detections:
        #     detections = (d[0], d[1], np.array(d[2]))

        detections_queue.put(detections)
        fps = int(1/(time.time() - prev_time))
        fps_queue.put(fps)
        # print("FPS: {}".format(fps))
        # darknet.print_detections(detections, args.ext_output)
    cap.release()




    # middleIdx = len(arr) // 2
    # firstVector = arr[middleIdx] - arr[0]
    # secondVector = arr[-1] - arr[middleIdx]
    # # print(secondVector, firstVector)
    # # import pdb
    # # pdb.set_trace()
    # sim = cosine_similarity([firstVector[:2]], [secondVector[:2]])
    # # print(sim)
    # if abs(secondVector[0] - firstVector[0]) > 15:
    #     sim = cosine_similarity([firstVector[:2]], [secondVector[:2]])

    #     if sim < _thres:
    #         return arr[middleIdx]
    # else:
    #     return None

# def test(img, arr, isShow=False):
#     p1 = (100,100)
#     p2 = (100,200)
#     p3 = (200,100)
#     for p in [p1,p2,p3]:
#         cv2.circle(img, p, 10, (0,0,255),-1)
#         cv2.putText(img, str(p), p, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
#     #print('arr:',arr)
#     if isShow:
#         cv2.imshow('test',img)
#         cv2.waitKey()



def check_mask(mask):
    while True:
        cv2.imshow('mask', mask)
        if cv2.waitKey(0) == 27:
            break


if __name__ == '__main__':
    args = parser()
    # ref_pts = np.load(r'C:\Users\OWNER\Documents\Jenny\20210323\points_arr.npy')
    watermark = r'C:\Users\OWNER\Documents\Jenny\210505\watermark.png'

    table_h = 456
    table_w = 822
    win_h = 672  # 556
    win_w = 1134  # 1022
    ndim = 2

    # name1 = ''
    name1 = 'NTCU'
    # name2 = ''  # nknu
    name2 = 'NKNU'
    
    ref_pts = np.load(args.ref_pts)
    t_fun = get_t_fun(ref_pts, table_h, table_w, win_h, win_w)

    _q_size = 1
    segRally_thres = 30
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=_q_size)
    detections_queue = Queue(maxsize=_q_size)
    fps_queue = Queue(maxsize=_q_size)
    frame_rgb_queue = Queue(maxsize=_q_size)
    time_queue = Queue(maxsize=_q_size)
    image_queue = Queue(maxsize=_q_size)
    tail_queue = Queue(maxsize=_q_size)
    hitPoint_queue = Queue(maxsize=_q_size)
    frame_queue_toShowing = Queue()
    pose_queue = Queue(maxsize=100)
    kp_queue = Queue(maxsize=30)
    hand_queue = Queue(maxsize=_q_size)
    left_handed = (False, False)

    report_hand_hit_queue = Queue(maxsize=_q_size)
    report_bounce_queue = Queue(maxsize=_q_size)

    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=1
    )
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)
    ret = cap.set(3, 1920)
    ret = cap.set(4, 1080)

    #mask = cv2.imread('/media/jennychen/DATA/experiment/210308/tableTennis/bounce_mask.jpg')
    if args.bounce_mask is None:
        mask = np.ones((1080, 1920, 3))
    else:
        mask = cv2.imread(args.bounce_mask)  # TODO
    if args.rotate != 0:
        mask = rotate(mask, args.rotate)

    if not args.draw_original:
        mask = cv2.resize(mask, (width, height),
                          interpolation=cv2.INTER_LINEAR)

    # bg_mask
    bg_mask = get_bg_mask(args.bg_mask)

    # skeleton mask
    skeleton_mask = cv2.imread('./imgs/skeleton_mask.png', cv2.IMREAD_GRAYSCALE)
    skeleton_mask = cv2.resize(skeleton_mask, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    # skeleton_mask = np.invert(skeleton_mask.astype(np.bool)).astype(np.uint8)

    # configure openpose
    params = dict()
    params["model_folder"] = "./models/"
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    watermark_img = cv2.imread(watermark)
    Thread(target=video_capture, args=(frame_queue, darknet_image_queue, frame_rgb_queue, pose_queue,
           args.fps_skip, time_queue, args.rotate, bg_mask, frame_queue_toShowing)).start()
    Thread(target=inference, args=(darknet_image_queue,
           detections_queue, fps_queue)).start()
    Thread(target=drawing, args=(cap, hitPoint_queue, frame_queue, detections_queue, fps_queue, mask, frame_rgb_queue,
           args.draw_original, time_queue, t_fun, segRally_thres, image_queue, tail_queue,
           report_hand_hit_queue, report_bounce_queue)).start()
    Thread(target=showing, args=(cap, width, height, args.show_2d, args.dont_show, args.out_filename, image_queue, hand_queue, mask, tail_queue, t_fun,
           args.draw_original, name1, name2, watermark_img, frame_queue_toShowing, report_hand_hit_queue, hitPoint_queue)).start()
    Thread(target=pose_estimate, args=(cap, op, opWrapper, pose_queue, kp_queue, skeleton_mask)).start()
    Thread(target=wait_for_hit_point, args=(cap, kp_queue, hitPoint_queue, hand_queue, left_handed)).start()
    Thread(target=report, args=(cap, report_hand_hit_queue, report_bounce_queue)).start()
    Thread(target=upload_report, args=(cap)).start()
