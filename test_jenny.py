import os
import cv2
import pdb
import sys
# sys.path.append('/media/jennychen/DATA/code/yolov4/table_tennis_fun')
import copy
import time
import random
import argparse
from ctypes import *
from queue import Queue
from threading import Thread, enumerate
import numpy as np
from numpy.core.defchararray import count
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from darknet import darknet
from fun_2d_new import *
from mask import generate_mask, apply_mask
from utils import calculate_angle

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
# Windows Import
# Change these variables to point to the correct folder (Release/x64 etc.)
sys.path.append(dir_path + '/testing/python/openpose/Release');
os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/testing/x64/Release;' +  dir_path + '/testing/bin;'
import pyopenpose as op

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

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
    parser.add_argument("--rotate", type=int, default= 0,
                        help="rotate image")
    parser.add_argument("--bounce_mask", type=str, default=None,
                        help="bounce mask image, <bounce_mask>.jpg")
    parser.add_argument("--draw_original",  action='store_true',
                        help="draw on original frame")
    parser.add_argument("--fps_skip", type=int, default= 1,
                        help="")
    parser.add_argument("--show_2d", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ref_pts", type=str, default = r'C:\Users\OWNER\Documents\Jenny\20210323\points_arr.npy', 
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
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
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

def video_capture(frame_queue, darknet_image_queue, frame_rgb_queue, fps_skip, time_queue, rotate_ang, bg_mask):
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        # if frame_count > start_frame:

        if (frame_count%fps_skip) == 0:
            # pose_queue.put((frame_count, frame))
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if rotate_ang != 0:
                frame_rgb = rotate(frame_rgb, rotate_ang) # rotate

            frame_resized = cv2.resize(frame_rgb, (width, height),
                                       interpolation=cv2.INTER_LINEAR)

            # frame_resized = cv2.putText(frame_resized, str(frame_count), (30, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
            #                     2, (0, 255, 255), 1, cv2.LINE_AA)
            frame_rgb_queue.put(frame_rgb)

            '''---'''
                        
            frame_queue.put(frame_resized)

            frame_resized_mask = copy.deepcopy(frame_resized)

            if not bg_mask is None:
                bg_mask = cv2.resize(bg_mask, (width, height),
                                           interpolation=cv2.INTER_LINEAR)
                # import pdb
                # pdb.set_trace()
                frame_resized_mask = frame_resized_mask*bg_mask
            
            darknet.copy_image_from_bytes(darknet_image, frame_resized_mask.tobytes())
            darknet_image_queue.put(darknet_image)
            time_queue.put(time.time())
        frame_count += 1
    cap.release()


def inference(darknet_image_queue, detections_queue):
    while cap.isOpened():
        darknet_image = darknet_image_queue.get()
        prev_time = time.time()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
        # for d in detections:
        #     detections = (d[0], d[1], np.array(d[2]))
        
        detections_queue.put(detections)
        fps = int(1/(time.time() - prev_time))
        # print("FPS: {}".format(fps))
        # darknet.print_detections(detections, args.ext_output)
    cap.release()

def draw_tail(detected_tail, image, _color=(255,255,0)): # rgb
    for bbox in detected_tail:
        if bbox is not None:
            left, top, right, bottom = bbox
            x,y,w,h = bbox
            # cv2.rectangle(image, (left, top), (right, bottom), _color, 1)
            # cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1) # TODO 半透明原型
            cv2.circle(image, (int(x),int(y)), 10, _color, -1)
 
    return image

def draw_hit_point(image, bboxes, _color=(226, 43, 138)):
    for x,y,w,h in bboxes:
        #cv2.ellipse(image, (int(x), int(y)), (8, 12), 90, 0, 360, _color, 1)
        cv2.ellipse(image, (int(x), int(y)), (4, 6), 90, 0, 360, _color, -1)
    return image

def draw_bounce(image, bboxes, _color=(0, 231, 255)):# r, g, b
    for x,y,w,h in bboxes:
        # cv2.ellipse(image, (int(x), int(y)), (4, 6), 75, 0, 360, _color, 1)
        # cv2.ellipse(image, (int(x), int(y)), (2, 3), 75, 0, 360, _color, -1)
        cv2.ellipse(image, (int(x), int(y)), (8, 12), 90, 0, 360, _color, 1)
        cv2.ellipse(image, (int(x), int(y)), (4, 6), 90, 0, 360, _color, -1)
    return image

def find_hit_point(_list):
    arr = np.array(_list)
    # print(arr)
    if np.sum(arr==None) > 0:
        arr = np.array([ np.array(element) for element in arr[arr!=None]])
    else:
        arr = np.array([ np.array(element) for element in arr])
    if len(arr)>3 :
        return get_delta_xy(arr, _thres=-0.5) # 10 #20210401

    return None    


def get_delta_xy(arr, _dis=-1, _thres=-0.9):
    middleIdx = len(arr) // 2
    firstVector = arr[middleIdx] - arr[0]
    secondVector = arr[-1] - arr[middleIdx]
    # print(secondVector, firstVector)
    if abs(secondVector[0] - firstVector[0]) > 8:
        sim = cosine_similarity([firstVector[:2]], [secondVector[:2]])
        if sim < _thres :
            return arr[middleIdx]
    else:
        return None


def find_y_min(_list):
    arr = np.array(_list)
    if np.sum(arr==None) > 0:
        arr = np.array([ np.array(element) for element in arr[arr!=None]])
    else:
        arr = np.array([ np.array(element) for element in arr])
    #if len(arr)>3 and deltaX_gt_deltaY(arr, _thres=10):
    if len(arr)>3 and deltaX_gt_thres(arr, _thres=15): # 10 #20210401
        arr_min_index, arr_min_value = find_min(arr[:,1])
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
    # if _diff[0] < 1000: # exclude max
    _delta = _diff[:-1]*_diff[1:]
    # print('_diff:{}\n_delta:{}'.format(_diff,_delta))
    if np.sum(_delta<0) > 0:
        min_index = np.argmin(_delta)+1
        min_value = arr[min_index]
        if min_value > arr[min_index+1]: # exclude max; In opencv coordinate is reverse so use '>'.
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
        self.bounce = [] # A score
        self.bounce_all = [] # A complete game
        self.hitPoint = []
        self.hitPoint_all = []
        self.hitPoint_range = {"up": [0,0,0,0,0], "down":[0,0,0,0,0]}
        self.text = ''
        self.mask = mask
        self.count = int(_size*0.5)
        self.isBounce = True
        self.hit_point_for_draw = None
        self.new_bounce = None
        self.prev_bounce = None
    def get_prev_bbox(self, _list):
        for b in _list:
            if b is not None:
                return b
        return None

    def get_closet_bbox(self, target, src_list,th=20):
        _min_value = 0
        cloest_bbox = None
        if target is None:
            return src_list[-1]
        else: 
            x,y,w,h = target
            dst_arr = np.array(src_list)
            _diff = dst_arr-target
            min_index = np.argmin(np.sum(_diff[:,:2]**2,axis=1))
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
    def inQ(self, bboxes):
        del self.detected[0]
        self.detected.append(bboxes) 
        self.valid_update(bboxes)
        self.arr_update()
        if self.count == 1:
            # hitPoint = find_hit_point(self.arr)
            # # with open(r'C:\Users\OWNER\Documents\tyler\tabletennis\hit_point.txt', 'a') as f:
            # #     f.write(str(hitPoint) + '\n')
            # if hitPoint is not None:
            #     hitPoint_queue.put(hitPoint[:2])
            #     self.hitPoint.append(hitPoint)
            #     get_hit_point_range(self.hitPoint_range, hitPoint, table_h, table_w, win_h, win_w, t_fun)
            #     self.hit_point_for_draw = hitPoint
            #     print(self.hitPoint_range)
            bounce = find_y_min(self.arr)
            if (bounce is not None) and (np.sum(self.mask[int(bounce[1]), int(bounce[0])]) > 0) \
                and np.sum(self.bounce==bounce)==0 and self.isBounce:
                # import pdb
                #pdb.set_trace()
                self.bounce.append(bounce)
                if self.new_bounce is not None:
                    self.prev_bounce = self.new_bounce
                self.new_bounce = bounce
                np.save('bounce',self.bounce)
            self.count = int(self._size*0.5)
        self.count -=1
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
        cv2.imshow('mask',mask)
        if cv2.waitKey(0) == 27:
                break

def drawing(frame_queue, detections_queue, mask, frame_rgb_queue, draw_original, time_queue, t_fun, segRally_thres, image_queue, tail_queue):
    noTail_count = 0
    random.seed(3)  # deterministic bbox colors
    tail = TAIL_ARR(6, mask)
    fps_total_list = []

    while cap.isOpened():
        frame_resized = frame_queue.get()
        detections = detections_queue.get()
        frame_rgb = frame_rgb_queue.get()        

        if draw_original:
            detections = map_point(frame_resized, frame_rgb, detections)#TODO
            frame_resized = frame_rgb
                         
        if frame_resized is not None: # there is the frame frame video.

            # put detection to tail
            if len(detections)>0:
                bboxes = []
                for label, confidence, bbox in detections:
                    bboxes.append(bbox)
                tail.inQ(bboxes)
            else:
                tail.inQ(None)
            
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
                    if len(tail.get_bounce_all())> 0:                                
                        np_bounce = np.array(tail.get_bounce())
                        # tail.bounce_all.append(tail.get_bounce())
                        # np.append(tail.bounce_all, tail.get_bounce(), axis=0)
                        # tail.bounce_all.append(np_bounce)
                        tail.bounce_all = np.append(tail.bounce_all, np_bounce, axis=0)
                    else:
                        # tail.bounce_all = tail.get_bounce()
                        np_bounce = np.array(tail.get_bounce())
                        tail.bounce_all = np_bounce

                # if len(tail.get_hit_point()) > 1:
                #     if len(tail.get_hit_point_all())> 0:                                
                #         np_hit_point = np.array(tail.get_hit_point())
                #         tail.hitPoint_all = np.append(tail.hitPoint_all, np_hit_point, axis=0)
                #     else:
                #         np_hit_point = np.array(tail.get_hit_point())
                #         tail.hitPoint_all = np_hit_point
                tail.clear_bounce() # clear bounces on a rally.
                tail.clear_hit_point()
            # cv2.putText(image, str(noTail_count), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 2, cv2.LINE_AA) # for check seg. rally
           
            if len(tail.get_hit_point()) > 0:
                # print(tail.get_hit_point())
                image = draw_hit_point(image, tail.get_hit_point())
                # np_hit_point = np.array(tail.get_hit_point())
                # tail.hitPoint = np_hit_point
                
            # draw bounce 
            if len(tail.get_bounce())>0:
                # pdb.set_trace()
                
                image = draw_bounce(image, tail.get_bounce())

                # cv2.imwrite('/media/jennychen/DATA/experiment/210303/tableTennis/bounce.png',image)
            # print(tail.get_bounce())
            #test(image,tail.get_bounce()) 
            # image = darknet.draw_boxes(detections, frame_resized, class_colors)
            # image = darknet.draw_boxes(detections, frame_resized, class_colors, show_text=False)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # bgr 2 rbg

            image_queue.put(image)
            tail_queue.put(tail)
        fps_total = 1/float(time.time()-time_queue.get())
        # print('\r isBounce:' tail.get_isBounce(), 'real_fps:',fps_total, 'bounce len:',len(tail.get_bounce()))
        print('\r rec: {}, real_fps: {}, bounce len:{}'.format(tail.get_isBounce(), fps_total, len(tail.get_bounce())), end='\r', flush=True) 
        
        fps_total_list.append(fps_total)
    cap.release()
    print('mean fps:',np.mean(np.array(fps_total_list)))

def changeNameSide(name1, name2):
    # change name
    name_temp = name1
    name1 = name2
    name2 = name_temp

    return name1, name2

def showing(out_filename, image_queue, hand_queue, mask, tail_queue, t_fun, draw_original, name1, name2, watermark_img):
    if draw_original:
        h, w,_ = mask.shape
        video = set_saved_video(cap, out_filename, (w, h))
    else:
        video = set_saved_video(cap, out_filename, (width, height))
    # img_demo =  draw_2d_demo(table_h, table_w, win_h, win_w, a_name=name1, b_name=name2)
    img_2d = draw_2d_bg(table_h, table_w, win_h, win_w, a_name=name1, b_name=name2)
    # temp_rectangle = np.zeros([90, 1920 , 3], dtype=np.uint8)
    # cv2.rectangle(temp_rectangle, (0, 0), (1920, 90), (204, 102, 0), -1)

    temp_rectangle = np.zeros([int(270/2), 1920 , 3], dtype=np.uint8)
    temp_rectangle_low = temp_rectangle.copy()
    # temp_rectangle_low = temp_rectangle_low + (watermark_img*0.5).astype(np.uint8)[:int(270/2),:]

    # cv2.rectangle(temp_rectangle, (0, 0), (1920, 270/2), (204, 102, 0), -1)

    # cv2.namedWindow("Inference", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("img_2d", cv2.WINDOW_NORMAL)

    stop_update = -1
    display_type = 1
    text_dict_h = False # false is for merge. true is for single
    img_hm = None
    cv2.namedWindow('full', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('full', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cc = 1
    hand = ''
    while cap.isOpened():
    
        image = image_queue.get()
        tail = tail_queue.get()
        # if not hand_queue.empty():
        #     hand = hand_queue.get()

        if image is not None: # there is the frame frame video.
            # ---------------
            import pdb
            #pdb.set_trace()
            # if out_filename is not None:
            #     video.write(image)
            
            if args.show_2d:
                # TODO 0504: Change tail.get_bounce()
                # np.reshape(np.array(tail.get_bounce_all()),(-1,4))
                # img_2d, trans_points = draw_2d(tail.get_bounce_all(), t_fun, img_2d_bg)
                # img_demo = draw_2d_realtime(tail.get_new_bounce(), tail.get_prev_bounce(), tail.get_hit_point_for_draw(), t_fun, img_demo)
                img_2d = draw_2d_realtime(tail.get_new_bounce(), tail.get_prev_bounce(), tail.get_hit_point_for_draw(), t_fun, img_2d)
                # cv2.imwrite(r'C:\Users\OWNER\Documents\tyler\tabletennis\output.jpg', img_2d)
                # video.write(img_2d)
                img_2d_90=np.rot90(img_2d)

            _wait_key = cv2.waitKey(1)

            if _wait_key == 27 or _wait_key == ord('r'): # clean bounce_rally
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

            elif _wait_key == ord('c'): # change side
                _time = str(time.time())
                # change_side(name1, name2, tail)
                name1, name2 = changeNameSide(name1, name2)
                # save 2d image file
                img_2d = draw_2d_realtime(tail.get_new_bounce(), tail.get_prev_bounce(), t_fun, img_2d)
                
                # img_2d, trans_points = draw_2d(tail.get_bounce(), t_fun, img_2d_bg)
                img_2d_90=np.rot90(img_2d)
                cv2.imwrite('img_2d_set_{}.png'.format(_time),img_2d)

                # save bounce
                np.save('bounce_set_{}'.format(_time),tail.get_bounce_all())
                
                # create new 2d bg
                img_2d = draw_2d_bg(table_h, table_w, win_h, win_w, a_name=name1, b_name=name2)

                # clean bounce
                tail.clear_bounce_all()


            elif _wait_key == ord('p'):

                if len(tail.get_bounce_all()) > 0:
                    stop_update *= -1
                    if stop_update == 1:
                        
                        if len(tail.get_bounce())>0:
                            np_bounce = np.array(tail.get_bounce())
                            bounce_all_now = np.append(tail.bounce_all, np_bounce, axis=0)
                        else:
                            bounce_all_now = np.array(tail.get_bounce_all())
                        frame_pts = np.reshape(bounce_all_now,(-1,4))

                        # trans_points = frameTo2D(frame_pts, t_fun)
                        
                        dst = get_table_corner_position(table_h, table_w, win_h, win_w)
                        bounce_pts = frameTo2D(frame_pts, t_fun)

                        img_hm = copy.deepcopy(img_2d)
                        x0, x1 = lower_uper(dst[:,0])
                        y0, y1 = lower_uper(dst[:,1])
                        cx, cy = (np.sum(dst,axis=0)/4 + 0.5).astype(int) # +0.5 is for round
                        img_hm, x_y_cunt_patchB = get_heatmap(img_2d, bounce_pts, 
                                                    x0, cx, y0, y1, color_rg='g', x_patch=2, y_patch=3)
                        img_hm, x_y_cunt_patchA = get_heatmap(img_hm, bounce_pts, 
                                                    cx, x1, y0, y1, color_rg='r', x_patch=2, y_patch=3)
                        # img_hm, x_y_cunt = get_heatmap(img_hm, bounce_pts, dst)
                        img_hm_90 = np.rot90(img_hm)
                        img_hm_90 = put_hm_text(img_hm_90, x_y_cunt_patchB)
                        img_hm_90 = put_hm_text(img_hm_90, x_y_cunt_patchA)

                        text_dict_h = False
                        import pdb
                        pdb.set_trace()
                        cv2.imwrite('imgs/temp/distribution.png', img_hm_90[439:1261, 276:732])

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

            if _wait_key in [49, 50, 51, 52, 53]: #1,2,3,4,5
                display_type = _wait_key - 48

            if not args.dont_show:
                # cv2.imshow('Inference', image)
                if stop_update == 1:
                    img_2d_90 = copy.deepcopy(img_hm_90)
                else:
                    img_2d_90=np.rot90(img_2d)
                # cv2.putText(image, str(stop_update), (50, 200 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # img_2d_90 = paint_chinese_opencv(img_2d_90, name1, (win_h//2-80, 20), (255, 255, 255))
                # img_2d_90 = paint_chinese_opencv(img_2d_90, name2, (win_h//2-80, win_w-60), (255, 255, 255))
                # img_2d_90=np.rot90(img_2d)
                if display_type in [1,2,3] :
                    if text_dict_h and (img_hm is not None) and stop_update == 1: # for check text dict
                        img_hm_90 = np.rot90(img_hm)
                        img_hm_90 = put_hm_text(img_hm_90, x_y_cunt_patchB)
                        img_hm_90 = put_hm_text(img_hm_90, x_y_cunt_patchA)
                        img_2d_90 = copy.deepcopy(img_hm_90)
                        text_dict_h = False 

                    img_merge = merge_image(image, img_2d_90, temp_rectangle, temp_rectangle_low, table_h, table_w, win_h, win_w)
                    # cv2.namedWindow('merge', cv2.WINDOW_NORMAL)
                    # cv2.setWindowProperty('merge', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    # cv2.imshow('full',img_merge)
                    show_img = img_merge.copy()

                if display_type in [4,5] :
                    if (not text_dict_h) and (img_hm is not None) and stop_update == 1: # for check text dict
                        # img_hm_90 = np.rot90(img_hm)
                        img_hm_copy = img_hm.copy()
                        img_hm_90 = put_hm_text_h(img_hm_copy, x_y_cunt_patchB)
                        img_hm_90 = put_hm_text_h(img_hm_copy, x_y_cunt_patchA)
                        # pdb.set_trace()
                        img_hm_90 = np.rot90(img_hm_90)
                        img_2d_90 = copy.deepcopy(img_hm_90)
                        
                        # pdb.set_trace()
                        text_dict_h = True

                    img_2d_360=np.rot90(img_2d_90, 3)
                    # cv2.imshow('full',img_2d_360)
                    show_img = img_2d_360.copy()
                    show_img = cv2.resize(show_img, (1920, 1080), interpolation=cv2.INTER_LINEAR)
                # import pdb
                # pdb.set_trace()
                # show_img = show_img + (watermark_img*0.5).astype(np.uint8)
                cc += 1

                # if cc >= 33*14:
                #     pdb.set_trace()
                cv2.putText(show_img, str(cc), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(show_img, 'Detect: {}'.format(hand), (800, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow('full',show_img)
                if out_filename is not None:
                    video.write(show_img)
                # if display_type == 5:
                #     img_2d_360=np.rot90(img_2d_90, 3)
                #     cv2.imshow('full',img_2d_360)

                    # TODO

    video.release()
    cv2.destroyAllWindows()
    cap.release()

def pose_estimate(op, opwrapper, pose_queue, kp_queue, skeleton_mask):
    while cap.isOpened():
        idx, frame = pose_queue.get()
        
        if frame is not None:
            masked_img = apply_mask(frame, skeleton_mask)            
            datum = op.Datum()
            datum.cvInputData = masked_img
            opwrapper.emplaceAndPop(op.VectorDatum([datum]))
            if datum.poseKeypoints is None:
                continue
            else:
                keypoints = datum.poseKeypoints[:2, 2:8, :-1]
            if kp_queue.full():
                kp_queue.get()
            kp_queue.put((idx, keypoints))
    cap.release()

def wait_for_hit_point(kp_queue, hitpoint_queue, hand_queue, left_handed):
    while cap.isOpened():
        if hitpoint_queue.empty():
            # time.sleep(0.01)
            continue
        else:
            hit_point = hitpoint_queue.get()
            print('hit', hit_point)
            is_forehand = judge_fore_back(kp_queue, hit_point, left_handed)
            is_forehand = 'forehand' if is_forehand else 'backhand'
            print(is_forehand)
            # hand_queue.put(is_forehand)
    
    cap.release()

def fore_or_back(angles):
    is_forehand = True
    diff = angles[-1] - angles[0]
    if diff >= 0:
        is_forehand = False    
    else:
        is_forehand = True

    return is_forehand

def judge_fore_back(kp_queue, hit_point, left_handed=(False, False)):
    # half_width = width / 2
    half_width = 1920 / 2
    is_left = hit_point[0] < half_width   # is the left person or not

    # which arm to be calculated
    is_left_handed = left_handed[0] if is_left else left_handed[1]
    arm_side = slice(3, 6) if is_left_handed else slice(0, 3)
    
    # get stored keypoints
    keypoints = []
    count = 0
    while not kp_queue.empty():
        keypoints.append(kp_queue.get())
        count += 1
    
    arm_kps = []
    count = 0
    for idx, points in keypoints:
        if len(points) < 2:
            continue
        count += 1
        if count >= 30:
            break
        # decide which person
        person = points[0]
        avg_x = (person[0][0] + person[1][0]) / 2
        right = avg_x > half_width
        if (not right and is_left) or (right and not is_left):
            arm_kps.append((idx, person[arm_side, :]))
        else:
            person = points[1]
            arm_kps.append((idx, person[arm_side, :]))
    # print(count)
    
    angles = []
    frame_idx = []
    for idx, arm in arm_kps:
        angle = calculate_angle(arm[0], arm[1], arm[2])
        angles.append(angle)
        frame_idx.append(idx)
    # print('angles', angles)
    if np.isnan(angles).any():
        s = pd.Series(angles)
        s = s.interpolate()
        angles = s.tolist()    

    angles = moving_average(angles, 5)
    # print('mv', angles)
    # print(frame_idx[0], frame_idx[-1])
    # print(len(angles))
    is_forhand = fore_or_back(angles)

    return is_forhand

if __name__ == '__main__':
    args = parser()
    # ref_pts = np.load(r'C:\Users\OWNER\Documents\Jenny\20210323\points_arr.npy')
    watermark = r'C:\Users\OWNER\Documents\Jenny\210505\watermark.png'

    table_h = 456
    table_w = 822
    win_h = int(672*1.5)#556
    win_w = int(1134*1.5)#1022
    ndim = 2

    name1 = 'FJU'
    # name1 = 'UT'
    name2 = 'NTUS'

    ref_pts = np.load(args.ref_pts)
    t_fun = get_t_fun(ref_pts, table_h, table_w, win_h, win_w)

    _q_size = 1
    segRally_thres = 30
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=_q_size)
    detections_queue = Queue(maxsize=_q_size)
    # fps_queue = Queue(maxsize=_q_size)
    frame_rgb_queue = Queue(maxsize=_q_size)
    time_queue = Queue(maxsize=_q_size)
    image_queue = Queue(maxsize=_q_size)
    tail_queue = Queue(maxsize=_q_size)
    pose_queue = Queue(maxsize=_q_size)
    kp_queue = Queue(maxsize=30)
    hitPoint_queue = Queue(maxsize=_q_size)
    hand_queue = Queue(maxsize=_q_size)

    left_handed=(False, False)
    
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
        mask = np.ones((1080,1920,3))
    else:
        mask = cv2.imread(args.bounce_mask)#TODO
    if args.rotate != 0:
        mask = rotate(mask, args.rotate)
    
    if not args.draw_original:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)

    # bg_mask
    bg_mask = get_bg_mask(args.bg_mask)

    # skeleton_mask = cv2.imread('./imgs/skeleton_mask.png', cv2.IMREAD_GRAYSCALE)
    # skeleton_mask = cv2.resize(skeleton_mask, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
     
    # Configure openpose
    # params = dict()
    # params["model_folder"] = "./models/"
    # params['num_gpu'] = 2
    # params['num_gpu_start'] = 1
    # opWrapper = op.WrapperPython()
    # opWrapper.configure(params)
    # opWrapper.start()

    watermark_img = cv2.imread(watermark)
    Thread(target=video_capture, args=(frame_queue, darknet_image_queue, frame_rgb_queue, args.fps_skip, time_queue, args.rotate, bg_mask)).start()
    Thread(target=inference, args=(darknet_image_queue, detections_queue)).start()
    # Thread(target=pose_estimate, args=(op, opWrapper, pose_queue, kp_queue, skeleton_mask)).start()
    Thread(target=drawing, args=(frame_queue, detections_queue, mask, frame_rgb_queue, args.draw_original, time_queue, t_fun, segRally_thres, image_queue, tail_queue)).start()
    Thread(target=showing, args=(args.out_filename, image_queue, hand_queue, mask, tail_queue, t_fun, args.draw_original, name1, name2, watermark_img)).start()
    # Thread(target=wait_for_hit_point, args=(kp_queue, hitPoint_queue, hand_queue, left_handed)).start()
