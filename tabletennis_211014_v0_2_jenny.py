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

from threading import Thread, enumerate
from sklearn.metrics.pairwise import cosine_similarity

import copy
from darknet import darknet
from fun_2d_ne import *


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


def set_saved_video(input_video, output_video, size):
    # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    # fps = int(input_video.get(cv2.CAP_PROP_FPS))
    fps = 30
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    import pdb
    # pdb.set_trace()
    return video


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


def video_capture(frame_queue, darknet_image_queue, frame_rgb_queue, fps_skip, time_queue, rotate_ang, bg_mask, frame_queue_toShowing):
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


def draw_tail(detected_tail, image, _color=(255, 255, 0)):  # rgb
    import cv2
    for bbox in detected_tail:
        if bbox is not None:
            left, top, right, bottom = bbox
            x, y, w, h = bbox
            # cv2.rectangle(image, (left, top), (right, bottom), _color, 1)
            # cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1) # TODO 半透明原型
            cv2.circle(image, (int(x), int(y)), 10, _color, -1)

    return image


def draw_hit_point(image, bboxes, _color=(226, 43, 138)):
    for x, y, w, h in bboxes:
        #cv2.ellipse(image, (int(x), int(y)), (8, 12), 90, 0, 360, _color, 1)
        cv2.ellipse(image, (int(x), int(y)), (4, 6), 90, 0, 360, _color, -1)
    return image


def draw_bounce(image, bboxes, _color=(0, 231, 255)):  # r, g, b
    for x, y, w, h in bboxes:
        # cv2.ellipse(image, (int(x), int(y)), (4, 6), 75, 0, 360, _color, 1)
        # cv2.ellipse(image, (int(x), int(y)), (2, 3), 75, 0, 360, _color, -1)
        cv2.ellipse(image, (int(x), int(y)), (8, 12), 90, 0, 360, _color, 1)
        cv2.ellipse(image, (int(x), int(y)), (4, 6), 90, 0, 360, _color, -1)
    return image


def find_hit_point(_list):
    arr = np.array(_list)
    # print(arr)
    if np.sum(arr == None) > 0:
        arr = np.array([np.array(element) for element in arr[arr != None]])
    else:
        arr = np.array([np.array(element) for element in arr])
    # print(arr)
    if len(arr) > 3 and deltaX_gt_thres(arr, _thres=15):
        min_idx = get_delta_xy(arr[:, 0], _thres=-0.3)  # 10 #20210401
        if min_idx is not None:
            return arr[min_idx]
    return None


def get_delta_xy(arr, _dis=-1, _thres=-0.9):
    
    _diff = arr[:-1]-arr[1:]
    _delta = _diff[:-1]*_diff[1:]
    if np.sum(_delta < 0) > 0:
        min_index = np.argmin(_delta)+1
       
        return min_index
    return None
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


def find_y_min(_list):
    arr = np.array(_list)
    if np.sum(arr == None) > 0:
        arr = np.array([np.array(element) for element in arr[arr != None]])
    else:
        arr = np.array([np.array(element) for element in arr])
    # if len(arr)>3 and deltaX_gt_deltaY(arr, _thres=10):
    if len(arr) > 3 and deltaX_gt_thres(arr, _thres=15):  # 10 #20210401
        arr_min_index, arr_min_value = find_min(arr[:, 1])
        if arr_min_index is not None:
            return arr[arr_min_index]
    return None


def deltaX_gt_thres(arr, _next_idx=-1, _thres=5):
    _diff = arr[0]-arr[_next_idx]
    # print('deltaX: ',abs(_diff[0]))
    return abs(_diff[0]) > _thres


def deltaX_gt_deltaY(arr, _dis=-1, _thres=50):
    _diff = arr[0]-arr[_dis]
    deltaX, deltaY = abs(_diff[0]), abs(_diff[1])
    # print('deltaX:',deltaX)
    # print('deltaY:',deltaY)
    if (deltaX - deltaY) > _thres:
        return True
    else:
        return False


def find_min(arr):
    _diff = arr[:-1]-arr[1:]
    #    if _diff[0] < 1000: # exclude max
    _delta = _diff[:-1]*_diff[1:]
    # print('_diff:{}\n_delta:{}'.format(_diff,_delta))
    if np.sum(_delta < 0) > 0:
        min_index = np.argmin(_delta)+1
        min_value = arr[min_index]
        # exclude max; In opencv coordinate is reverse so use '>'.
        if min_value > arr[min_index+1]:
            return min_index, min_value
    return None, None

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


class TAIL_ARR():
    def __init__(self, _size, mask):
        self._size = _size
        self.detected = [None for i in range(_size)]
        self.detected_valid = [None for i in range(_size)]
        self.arr = [None for i in range(_size)]
        self.bounce = []  # A score
        self.bounce_all = []  # A complete game
        self.hitPointCount = 4
        self.hitPoint = []
        self.hitPoint_all = []
        self.hitPoint_range = {"up": [0, 0, 0, 0, 0], "down": [0, 0, 0, 0, 0]}
        self.text = ''
        self.mask = mask
        self.hit_point_for_draw = None
        self.count = int(_size*0.5)
        self.isBounce = True
        self.new_bounce = None
        self.prev_bounce = None
        self.hitPoint_range_num = -1

    def get_prev_bbox(self, _list):
        for b in _list:
            if b is not None:
                return b
        return None

    def get_closet_bbox(self, target, src_list, th=20):
        _min_value = 0
        cloest_bbox = None
        if target is None:
            return src_list[-1]
        else:
            x, y, w, h = target
            dst_arr = np.array(src_list)
            _diff = dst_arr-target
            min_index = np.argmin(np.sum(_diff[:, :2]**2, axis=1))
            return src_list[min_index]

    def valid_update(self, bboxes):
        if bboxes is not None:
            prev_bbox = self.get_prev_bbox(self.detected_valid)
            bbox = self.get_closet_bbox(prev_bbox, bboxes)
            self.detected_valid.append(bbox)
        else:
            self.detected_valid.append(None)

        del self.detected_valid[0]

    def arr_update(self):
        # TODO 補線
        self.arr = copy.deepcopy(self.detected_valid)
    def inQ(self, bboxes, report_bounce_queue, report_hand_hit_queue):
        del self.detected[0]
        self.detected.append(bboxes)
        self.valid_update(bboxes)
        self.arr_update()

        if self.count == 1:
            if self.hitPointCount % 4 == 0:
                
                hitPoint = find_hit_point(self.arr)
            else:
                hitPoint = None
                if self.hitPointCount != 4:
                    self.hitPointCount += 1

               
            if hitPoint is not None \
                and (report_hand_hit_queue.empty()) \
                and (np.abs(t_fun(hitPoint[:2])[0][0] - win_w/2) > 50): 
                # (report_hand_hit_queue.empty()) 為了解決deadlock, 但如果前一的 bounce miss, 會造成這次的hit被跳過
                # (np.abs(t_fun(hitPoint[:2])[0][0] - win_w/2) > 50) 濾掉打到網子的 離網子太近的不計
                
                hitPoint_queue.put(hitPoint[:2])
                self.hitPointCount = 1
                self.hitPoint.append(hitPoint)
                self.hitPoint_range_num = get_hit_point_range(
                    self.hitPoint_range, hitPoint, table_h, table_w, win_h, win_w, t_fun)
                self.hit_point_for_draw = hitPoint
           
            bounce = find_y_min(self.arr)
            
            if (bounce is not None) and (np.sum(self.mask[int(bounce[1]), int(bounce[0])]) > 0) \
                    and np.sum(self.bounce == bounce) == 0 and self.isBounce:
                import pdb
                # pdb.set_trace()
                self.bounce.append(bounce)
                # print('line 375')
                if self.new_bounce is not None:
                    # print('self.new_bounce is not None')
                    self.prev_bounce = self.new_bounce
                    if not report_hand_hit_queue.empty():
                        report_bounce_queue.put(bounce)
                self.new_bounce = bounce
                np.save('bounce', self.bounce)
            
            self.count = int(self._size*0.5)
        self.count -= 1
        # elif bounce is not None: # bug
        #     import pdb
        #     # pdb.set_trace()

    def get_arr(self):
        return self.arr

    def get_hit_point(self):
        return self.hitPoint

    def get_hit_point_all(self):
        return self.hitPoint_all

    def get_hit_point_for_draw(self):
        return self.hit_point_for_draw

    def clear_hit_point(self):
        self.hitPoint = []

    def get_bounce(self):
        return self.bounce

    def clear_bounce(self):
        self.bounce = []

    def stop_bounce(self):
        self.isBounce = False

    def rec_bounce(self):
        self.isBounce = True

    def get_isBounce(self):
        return str(self.isBounce)

    def get_prev_bounce(self):
        return self.prev_bounce

    def get_new_bounce(self):
        return self.new_bounce

    def get_bounce_all(self):
        return self.bounce_all

    def clear_bounce_all(self):
        self.bounce_all = []
        self.bounce = []
        self.new_bounce = None
        self.prev_bounce = None
    
    def get_hitpoint_range_num(self):
        return self.hitPoint_range_num


def map_point(src_img, dst_img, src_pts):
    import pdb
    dst_pts = []
    src_h, src_w, _ = src_img.shape
    dst_h, dst_w, _ = dst_img.shape
    w_scale = dst_w/src_w
    h_scale = dst_h/src_h
    for i in src_pts:
        x, w = i[2][0]*w_scale, i[2][2]*w_scale
        y, h = i[2][1]*h_scale, i[2][2]*h_scale
        dst_pts.append((i[0], i[1], (x, y, w, h)))
    return dst_pts


def check_mask(mask):
    while True:
        cv2.imshow('mask', mask)
        if cv2.waitKey(0) == 27:
            break

def drawing(frame_queue, detections_queue, fps_queue, mask, frame_rgb_queue, draw_original, 
    time_queue, t_fun, segRally_thres, image_queue, tail_queue, report_hand_hit_queue, report_bounce_queue):
    noTail_count = 0    
    import time
    random.seed(3)  # deterministic bbox colors
    tail = TAIL_ARR(6, mask)
    fps_total_list = []

    while cap.isOpened():
        frame_resized = frame_queue.get()
        detections = detections_queue.get()
        frame_rgb = frame_rgb_queue.get()
        fps = fps_queue.get()
        if draw_original:
            detections = map_point(
                frame_resized, frame_rgb, detections)  # TODO
            frame_resized = frame_rgb
        if frame_resized is not None:  # there is the frame frame video.

            # put detection to tail
            if len(detections) > 0:
                bboxes = []
                for label, confidence, bbox in detections:
                    bboxes.append(bbox)
                tail.inQ(bboxes, report_bounce_queue, report_hand_hit_queue)
            else:
                tail.inQ(None, report_bounce_queue, report_hand_hit_queue)
            
            # draw tail
            image = draw_tail(tail.get_arr(), frame_resized)

            # seg. rally
            if np.sum(np.array(tail.get_arr()) != None) < 3:
                noTail_count += 1
            else:
                noTail_count = 0
            if noTail_count > segRally_thres:
                # cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
                # text = 'Seg. Rally'
                # cv2.putText(image, text, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 2, cv2.LINE_AA)

                # TODO clean frame
                if len(tail.get_bounce()) > 1:
                    if len(tail.get_bounce_all()) > 0:
                        np_bounce = np.array(tail.get_bounce())
                        # tail.bounce_all.append(tail.get_bounce())
                        # np.append(tail.bounce_all, tail.get_bounce(), axis=0)
                        # tail.bounce_all.append(np_bounce)
                        tail.bounce_all = np.append(
                            tail.bounce_all, np_bounce, axis=0)
                    else:
                        # tail.bounce_all = tail.get_bounce()
                        np_bounce = np.array(tail.get_bounce())
                        tail.bounce_all = np_bounce

                    if len(tail.get_hit_point()) > 1:
                        if len(tail.get_hit_point_all()) > 0:
                            np_hit_point = np.array(tail.get_hit_point())
                            tail.hitPoint_all = np.append(
                                tail.hitPoint_all, np_hit_point, axis=0)
                        else:
                            np_hit_point = np.array(tail.get_hit_point())
                            tail.hitPoint_all = np_hit_point
                tail.clear_bounce()  # clear bounces on a rally.
                tail.clear_hit_point()

                # clean report queue
                if not report_bounce_queue.empty():
                    _ = report_bounce_queue.get()
                if not report_hand_hit_queue.empty():
                    _ = report_hand_hit_queue.get()

            # cv2.putText(image, str(noTail_count), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 2, cv2.LINE_AA) # for check seg. rally
            if len(tail.get_hit_point()) > 0:
                # print(tail.get_hit_point())
                image = draw_hit_point(image, tail.get_hit_point())
                # np_hit_point = np.array(tail.get_hit_point())
                # tail.hitPoint = np_hit_point
            # draw bounce
            if len(tail.get_bounce()) > 0:
                # pdb.set_trace()

                image = draw_bounce(image, tail.get_bounce())

                # cv2.imwrite('/media/jennychen/DATA/experiment/210303/tableTennis/bounce.png',image)
            # print(tail.get_bounce())
            # test(image,tail.get_bounce())
            # image = darknet.draw_boxes(detections, frame_resized, class_colors)
            # image = darknet.draw_boxes(detections, frame_resized, class_colors, show_text=False)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # bgr 2 rbg

            image_queue.put(image)
            tail_queue.put(tail)
        fps_total = 1/float(time.time()-time_queue.get())
        # print('\r isBounce:' tail.get_isBounce(), 'real_fps:',fps_total, 'bounce len:',len(tail.get_bounce()))
        print('\r rec: {}, real_fps: {}, bounce len:{}'.format(
            tail.get_isBounce(), fps_total, len(tail.get_bounce())), end='\r', flush=True)

        fps_total_list.append(fps_total)
    cap.release()
    print('mean fps:', np.mean(np.array(fps_total_list)))


def changeNameSide(name1, name2):
    # change name
    name_temp = name1
    name1 = name2
    name2 = name_temp

    return name1, name2


def showing(out_filename, image_queue, mask, tail_queue, t_fun, draw_original, name1, name2, 
    watermark_img, frame_queue_toShowing, report_hand_hit_queue, hitPoint_queue):
    if draw_original:
        h, w, _ = mask.shape
        video = set_saved_video(cap, out_filename, (w, h))
    else:
        video = set_saved_video(cap, out_filename, (width, height))

    img_2d = draw_2d_bg(table_h, table_w, win_h, win_w,
                        a_name=name1, b_name=name2)
    # temp_rectangle = np.zeros([90, 1920 , 3], dtype=np.uint8)
    # cv2.rectangle(temp_rectangle, (0, 0), (1920, 90), (204, 102, 0), -1)

    temp_rectangle = np.zeros([int(270/2), 1920, 3], dtype=np.uint8)
    temp_rectangle_low = temp_rectangle.copy()
    temp_rectangle_low = temp_rectangle_low + \
        (watermark_img*0.5).astype(np.uint8)[:int(270/2), :]

    # cv2.rectangle(temp_rectangle, (0, 0), (1920, 270/2), (204, 102, 0), -1)

    # cv2.namedWindow("Inference", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("img_2d", cv2.WINDOW_NORMAL)

    stop_update = -1
    display_type = 1
    text_dict_h = False  # false is for merge. true is for single
    img_hm = None
    cv2.namedWindow('full', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('full', cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    hand = ''
    while cap.isOpened():
       show_img_ori = frame_queue_toShowing.get()
       image = image_queue.get()
       tail = tail_queue.get()
       if image is not None: # there is the frame frame video.
            if not hitPoint_queue.empty():
                hitPoint_queue.get()
                # import pdb
                
                # print('612 report_bounce_queue',report_bounce_queue.qsize(),' report_hand_hit_queue',report_hand_hit_queue.qsize())
                # pdb.set_trace()
           # TODO 211014 if not hand_queue.empty():
           #    hand = hand_queue.get()
           #    # jenny
           #    if hand =='forehand':
           #         hand_num = 0
           #    else:
           #         hand_num = 1
                hand_num = 2 # no hand info.
                # print('--before report_hand_hit_queue put\n')
                report_hand_hit_queue.put([hand_num, tail.hitPoint_range_num])
                # print('--after report_hand_hit_queue put\n')

            # ---------------
            # pdb.set_trace()
            # if out_filename is not None:
            #     video.write(image)

            if args.show_2d:
                # TODO 0504: Change tail.get_bounce()
                # np.reshape(np.array(tail.get_bounce_all()),(-1,4))
                # img_2d, trans_points = draw_2d(tail.get_bounce_all(), t_fun, img_2d_bg)
                img_2d = draw_2d_realtime(tail.get_new_bounce(), tail.get_prev_bounce(
                ), tail.get_hit_point_for_draw(), t_fun, img_2d, tail.get_hitpoint_range_num())
                img_2d_90 = np.rot90(img_2d)

            _wait_key = cv2.waitKey(1)

            if _wait_key == 27 or _wait_key == ord('r'):  # clean bounce_rally
                _time = str(time.time())
                # np.save('bounce_{}'.format(_time),tail.get_bounce())
                # img_2d = draw_2d_realtime(tail.get_new_bounce(), tail.get_prev_bounce(), t_fun, img_2d)
                # # img_2d, trans_points = draw_2d(tail.get_bounce(), t_fun, img_2d_bg)
                # img_2d_90=np.rot90(img_2d)
                # cv2.imwrite('img_2d_{}.png'.format(_time),img_2d)

                tail.clear_bounce()

                # img_2d = draw_2d_bg(table_h, table_w, win_h, win_w, a_name=name1, b_name=name2)
                # cv2.imshow('img_2d',img_2d_90)

                if _wait_key == 27:
                    break
            # elif args.show_2d:
            #     img_2d, trans_points = draw_2d(tail.get_bounce(), t_fun, img_2d_bg)

            #     img_2d_90=np.rot90(img_2d)
                # cv2.imshow('img_2d',img_2d_90)

            elif _wait_key == ord('c'):  # change side
                _time = str(time.time())
                # change_side(name1, name2, tail)
                name1, name2 = changeNameSide(name1, name2)
                # save 2d image file
                img_2d = draw_2d_realtime(
                    tail.get_new_bounce(), tail.get_prev_bounce(), t_fun, img_2d)

                # img_2d, trans_points = draw_2d(tail.get_bounce(), t_fun, img_2d_bg)
                img_2d_90 = np.rot90(img_2d)
                cv2.imwrite('img_2d_set_{}.png'.format(_time), img_2d)

                # save bounce
                np.save('bounce_set_{}'.format(_time), tail.get_bounce_all())

                # create new 2d bg
                img_2d = draw_2d_bg(table_h, table_w, win_h,
                                    win_w, a_name=name1, b_name=name2)

                # clean bounce
                tail.clear_bounce_all()

            elif _wait_key == ord('p'):

                # if len(tail.get_bounce_all()) > 0:
                #     stop_update *= -1
                #     if stop_update == 1:

                # stop_update *= -1
                # if stop_update == 1 and len(tail.get_bounce_all()) > 0:

                if len(tail.get_bounce_all()) > 0:
                    stop_update *= -1
                    if stop_update == 1:

                        if len(tail.get_bounce()) > 0:
                            np_bounce = np.array(tail.get_bounce())
                            bounce_all_now = np.append(
                                tail.bounce_all, np_bounce, axis=0)
                        else:
                            bounce_all_now = np.array(tail.get_bounce_all())
                        frame_pts = np.reshape(bounce_all_now, (-1, 4))

                        # trans_points = frameTo2D(frame_pts, t_fun)

                        dst = get_table_corner_position(
                            table_h, table_w, win_h, win_w)
                        bounce_pts = frameTo2D(frame_pts, t_fun)

                        img_hm = copy.deepcopy(img_2d)
                        x0, x1 = lower_uper(dst[:, 0])
                        y0, y1 = lower_uper(dst[:, 1])
                        # +0.5 is for round
                        cx, cy = (np.sum(dst, axis=0)/4 + 0.5).astype(int)
                        img_hm, x_y_cunt_patchB = get_heatmap(img_2d, bounce_pts,
                                                              x0, cx, y0, y1, color_rg='g', x_patch=2, y_patch=3)
                        img_hm, x_y_cunt_patchA = get_heatmap(img_hm, bounce_pts,
                                                              cx, x1, y0, y1, color_rg='r', x_patch=2, y_patch=3)
                        # img_hm, x_y_cunt = get_heatmap(img_hm, bounce_pts, dst)
                        img_hm_90 = np.rot90(img_hm)
                        img_hm_90 = put_hm_text(img_hm_90, x_y_cunt_patchB)
                        img_hm_90 = put_hm_text(img_hm_90, x_y_cunt_patchA)

                        text_dict_h = False

                        # prob_map = probability_map(trans_points, ndim, table_h, table_w, win_h, win_w)
                        # prob_map = np.rot90(prob_map)
                        # prob_map_img = get_heatmap(prob_map, table_h, table_w, win_h, win_w)
                        # prob_map_img = prob_map_img[100:prob_map.shape[0]-300, 280:prob_map.shape[1]-100]
                        # img_2d_90 = add_img_hm(img_2d_90, prob_map_img, table_h, table_w, win_h, win_w)
                        # stop_update_img = img_hm.copy()
            elif _wait_key == ord('s'):
                tail.stop_bounce()
            elif _wait_key == ord('o'):
                tail.rec_bounce()

            if _wait_key in [49, 50, 51, 52, 53, 55]:  # 1,2,3,4,5,7
                display_type = _wait_key - 48

            if not args.dont_show:
                # cv2.imshow('Inference', image)
                if stop_update == 1:
                    img_2d_90 = copy.deepcopy(img_hm_90)
                else:
                    img_2d_90 = np.rot90(img_2d)
                # cv2.putText(image, str(stop_update), (50, 200 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # img_2d_90 = paint_chinese_opencv(img_2d_90, name1, (win_h//2-80, 20), (255, 255, 255))
                # img_2d_90 = paint_chinese_opencv(img_2d_90, name2, (win_h//2-80, win_w-60), (255, 255, 255))
                # img_2d_90=np.rot90(img_2d)
                if display_type in [1, 2, 3]:
                    # for check text dict
                    if text_dict_h and (img_hm is not None) and stop_update == 1:
                        img_hm_90 = np.rot90(img_hm)
                        img_hm_90 = put_hm_text(img_hm_90, x_y_cunt_patchB)
                        img_hm_90 = put_hm_text(img_hm_90, x_y_cunt_patchA)
                        img_2d_90 = copy.deepcopy(img_hm_90)
                        text_dict_h = False

                    img_merge = merge_image(
                        image, img_2d_90, temp_rectangle, temp_rectangle_low, table_h, table_w, win_h, win_w)
                    # cv2.namedWindow('merge', cv2.WINDOW_NORMAL)
                    # cv2.setWindowProperty('merge', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    # cv2.imshow('full',img_merge)
                    show_img = img_merge.copy()

                if display_type in [4, 5]:
                    if (not text_dict_h) and (img_hm is not None) and stop_update == 1:  # for check text dict
                        # img_hm_90 = np.rot90(img_hm)
                        img_hm_copy = img_hm.copy()
                        img_hm_90 = put_hm_text_h(img_hm_copy, x_y_cunt_patchB)
                        img_hm_90 = put_hm_text_h(img_hm_copy, x_y_cunt_patchA)
                        # pdb.set_trace()
                        img_hm_90 = np.rot90(img_hm_90)
                        img_2d_90 = copy.deepcopy(img_hm_90)

                        # #
                        # cv2.putText(img_2d, '成大資工 多媒體與電腦視覺實驗室', (100, 50), f,
                        #     1, (255, 255, 255), 2, cv2.LINE_AA)

                        # pdb.set_trace()
                        text_dict_h = True

                    img_2d_360 = np.rot90(img_2d_90, 3)
                    # cv2.imshow('full',img_2d_360)
                    show_img = img_2d_360.copy()
                    show_img = cv2.resize(
                        show_img, (1920, 1080), interpolation=cv2.INTER_LINEAR)
                if display_type == 7:
                    show_img = cv2.cvtColor(show_img_ori, cv2.COLOR_BGR2RGB)
                import pdb
                # pdb.set_trace()
                # show_img = show_img + (watermark_img*0.5).astype(np.uint8)\

                cv2.imshow('full', show_img)
                if out_filename is not None:
                    video.write(show_img)
                # if display_type == 5:
                #     img_2d_360=np.rot90(img_2d_90, 3)
                #     cv2.imshow('full',img_2d_360)

                # TODO

    video.release()
    cv2.destroyAllWindows()
    cap.release()

def report(report_hand_hit_queue, report_bounce_queue):
    report_array = []

    import pyrebase
    config = {
        "apiKey": "AIzaSyAR1TLj8zXo4iLjlUAXS2fpacIqcc9dFUg",
        "authDomain": "volleyai.firebaseapp.com",
        "projectId": "volleyai",
        "storageBucket": "volleyai.appspot.com",
        "messagingSenderId": "412993216154",
        "appId": "1:412993216154:web:35daf1ea438d54a42bc923",
        "measurementId": "G-3Q59CXEPT9",
        "serviceAccount": "serviceAccount.json",
        "databaseURL": "https://volleyai-default-rtdb.asia-southeast1.firebasedatabase.app/"
    }

    firebase = pyrebase.initialize_app(config)

    # get firebase storage instance
    storage = firebase.storage()

    while cap.isOpened():
        print('report_bounce_queue',report_bounce_queue.qsize(),' report_hand_hit_queue',report_hand_hit_queue.qsize())
        bounce = report_bounce_queue.get() # report_bounce_queue.get() 要在 report_hand_hit_queue.get() 前面
        hand_num, hitPoint_range_num = report_hand_hit_queue.get()
        report_array.append([hand_num, hitPoint_range_num, bounce[0], bounce[1]])
        #pdb.set_trace()



        if len(report_array) > 0:
            img_2d = draw_2d_bg(table_h, table_w, win_h, win_w, a_name=name1, b_name=name2)
            dst = get_table_corner_position(table_h, table_w, win_h, win_w)
            # bounce_pts = frameTo2D(frame_pts, t_fun)
            bounce_pts = t_fun(np.array(report_array)[:,-2:])
            x0, x1 = lower_uper(dst[:,0])
            y0, y1 = lower_uper(dst[:,1])
            cx, cy = (np.sum(dst,axis=0)/4 + 0.5).astype(int) # +0.5 is for round
            
            _hand_key_arr = ['positive','negtive','pos_neg']
            _dict_side = dict()
            _dict_side_hit = dict() # for bounce_region.json
            

            _side_key_arr = ['left', 'right']
            _key_arr = ['one','two','three','four','five','one','two','three','four','five']

            report_np = np.array(report_array)

            img_hm = copy.deepcopy(img_2d)
            for j in [0,5]: # side
                _dict_hand = dict()
                _dict_hand_hit = dict()
                _bounce_side = (report_np[:,1]>=j)*(report_np[:,1]<j+5) 

                for h in [0,1,2]: # hand
                    _dict = dict()
                    
                    if h == 2:
                        _bounce_hand = np.ones_like(report_np[:,0]).astype('bool')
                        import pdb
                        # pdb.set_trace()

                    else:
                        _bounce_hand = (report_np[:,0]==h)
                    

                    for i in range(j, j+5): # hit
                        _bounce_pts = bounce_pts[(report_np[:,1]==i) * _bounce_side * _bounce_hand]

                        if j==5 :
                            img_hm, x_y_cunt_patch = get_heatmap(img_2d, _bounce_pts, 
                                                    x0, cx, y0, y1, color_rg='g', x_patch=2, y_patch=3, returnImg=False) # patchB
                        else:
                            img_hm, x_y_cunt_patch = get_heatmap(img_hm, _bounce_pts, 
                                                    cx, x1, y0, y1, color_rg='r', x_patch=2, y_patch=3, returnImg=False) # patchA

                        # img_hm = put_hm_text_h(img_hm, x_y_cunt_patch)
                        _percentage = np.array(x_y_cunt_patch)[:,-1]
                        if np.sum(_percentage)!=0:
                            _percentage = (_percentage / np.sum(_percentage))*100
                        # print('_percentage:', _percentage)
                        print('i:',i)
                        _dict[_key_arr[i]]=[str(int(_p+0.5)) for _p in _percentage] 
                    
                    # all
                    _bounce_pts = bounce_pts[_bounce_hand*_bounce_side]
                    if j == 5:
                        img_hm, x_y_cunt_patch = get_heatmap(img_2d, _bounce_pts, 
                                                    x0, cx, y0, y1, color_rg='g', x_patch=2, y_patch=3, returnImg=False) # patchB
                    else:
                        img_hm, x_y_cunt_patch = get_heatmap(img_hm, _bounce_pts, 
                                                    cx, x1, y0, y1, color_rg='r', x_patch=2, y_patch=3, returnImg=False) # patchA
                    _percentage = np.array(x_y_cunt_patch)[:,-1]
                    if np.sum(_percentage)!=0:
                        _percentage = (_percentage / np.sum(_percentage))*100
                    _dict['all']=[str(int(_p+0.5)) for _p in _percentage] 

                    # put to hand
                    _dict_hand[_hand_key_arr[h]] = _dict

                    # for bounce_region.json
                    
                    import pdb
                    _percentage_hit = report_np[_bounce_hand*_bounce_side][:,1]
                    # if np.sum(_percentage_hit)!=0:
                    if len(_percentage_hit)!=0:
                        _percentage_hit = (np.histogram(report_np[_bounce_hand][:,1], bins=10,range=(0,10)))[0]
                    else:
                        _percentage_hit = np.zeros([1,10])[0]
                    
                    if j==0:
                        _percentage_hit = _percentage_hit[:5]
                    else:
                        _percentage_hit = _percentage_hit[-5:]
                    if np.sum(_percentage_hit)!=0:
                        _percentage_hit = (_percentage_hit/np.sum(_percentage_hit)) *100
                        
                    _dict_hand_hit[_hand_key_arr[h]] = [str(int(_p+0.5)) for _p in _percentage_hit]
                
                _pos_hand = np.sum(report_np[_bounce_side][:,0]==0) # forehand
                _neg_hand = np.sum(report_np[_bounce_side][:,0]==1) # backhead

                if (_pos_hand+_neg_hand) != 0:
                    _hand_ratio = [_pos_hand, _neg_hand] / (_pos_hand+_neg_hand)
                else:
                    _hand_ratio = [0,0]

                _dict_hand_hit['ratio'] = [str(int(_p+0.5)) for _p in _hand_ratio]



                # put to side
                if j == 5:
                    _dict_side[_side_key_arr[1]] = _dict_hand
                    _dict_side_hit[_side_key_arr[1]] =_dict_hand_hit 
                else:
                    _dict_side[_side_key_arr[0]] = _dict_hand
                    _dict_side_hit[_side_key_arr[0]] =_dict_hand_hit
                    
            import json
            json_name = 'hit_region.json'
            with open(json_name, 'w') as f:
                json.dump(_dict_side, f)
            storage.child(json_name).put(json_name)

            json_name = 'bounce_region.json'
            with open(json_name, 'w') as f:
                json.dump(_dict_side_hit, f)
            storage.child(json_name).put(json_name)
            # import pdb
            # pdb.set_trace()
            # print('---')

            
                #-----#
    cap.releace()
if __name__ == '__main__':
    args = parser()
    # ref_pts = np.load(r'C:\Users\OWNER\Documents\Jenny\20210323\points_arr.npy')
    watermark = r'C:\Users\OWNER\Documents\Jenny\210505\watermark.png'

    table_h = 456
    table_w = 822
    win_h = 672  # 556
    win_w = 1134  # 1022
    ndim = 2

    name1 = ''
    # name1 = 'NTCU'
    name2 = ''  # nknu

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

    watermark_img = cv2.imread(watermark)
    Thread(target=video_capture, args=(frame_queue, darknet_image_queue, frame_rgb_queue,
           args.fps_skip, time_queue, args.rotate, bg_mask, frame_queue_toShowing)).start()
    Thread(target=inference, args=(darknet_image_queue,
           detections_queue, fps_queue)).start()
    Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue, mask, frame_rgb_queue, args.draw_original, time_queue, t_fun, segRally_thres, image_queue, tail_queue, 
        report_hand_hit_queue, report_bounce_queue)).start()
    Thread(target=showing, args=(args.out_filename, image_queue, mask, tail_queue, t_fun,
           args.draw_original, name1, name2, watermark_img, frame_queue_toShowing, report_hand_hit_queue, hitPoint_queue)).start()
    Thread(target=report, args=(report_hand_hit_queue,report_bounce_queue)).start()
