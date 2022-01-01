import numpy as np
import copy

from fun_2d_ne import *

table_h = 456
table_w = 822
win_h = 672  # 556
win_w = 1134  # 1022
ndim = 2

name1 = 'NTCU'
name2 = 'NKNU'

ref_pts = np.load(r'C:\Users\OWNER\Documents\Jenny\20210323\points_arr.npy')
t_fun = get_t_fun(ref_pts, table_h, table_w, win_h, win_w)

class TraceRecord():
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

    def deltaX_gt_thres(self, arr, _next_idx=-1, _thres=5):
        _diff = arr[0]-arr[_next_idx]
        # print('deltaX: ',abs(_diff[0]))
        return abs(_diff[0]) > _thres


    def deltaX_gt_deltaY(self, arr, _dis=-1, _thres=50):
        _diff = arr[0]-arr[_dis]
        deltaX, deltaY = abs(_diff[0]), abs(_diff[1])
        # print('deltaX:',deltaX)
        # print('deltaY:',deltaY)
        if (deltaX - deltaY) > _thres:
            return True
        else:
            return False


    def find_min(self, arr):
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

    def find_hit_point(self, _list):
        arr = np.array(_list)
        # print(arr)
        if np.sum(arr == None) > 0:
            arr = np.array([np.array(element) for element in arr[arr != None]])
        else:
            arr = np.array([np.array(element) for element in arr])
        # print(arr)
        if len(arr) > 3 and self.deltaX_gt_thres(arr, _thres=15):
            min_idx = self.get_delta_xy(arr[:, 0], _thres=-0.3)  # 10 #20210401
            if min_idx is not None:
                return arr[min_idx]
        return None

    def get_delta_xy(self, arr, _dis=-1, _thres=-0.9):
        
        _diff = arr[:-1]-arr[1:]
        _delta = _diff[:-1]*_diff[1:]
        if np.sum(_delta < 0) > 0:
            min_index = np.argmin(_delta)+1
        
            return min_index
        return None

    def find_y_min(self, _list):
        arr = np.array(_list)
        if np.sum(arr == None) > 0:
            arr = np.array([np.array(element) for element in arr[arr != None]])
        else:
            arr = np.array([np.array(element) for element in arr])
        # if len(arr)>3 and deltaX_gt_deltaY(arr, _thres=10):
        if len(arr) > 3 and self.deltaX_gt_thres(arr, _thres=15):  # 10 #20210401
            arr_min_index, arr_min_value = self.find_min(arr[:, 1])
            if arr_min_index is not None:
                return arr[arr_min_index]
        return None
    
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
    def inQ(self, hitPoint_queue, bboxes, report_bounce_queue, report_hand_hit_queue):
        del self.detected[0]
        self.detected.append(bboxes)
        self.valid_update(bboxes)
        self.arr_update()

        if self.count == 1:
            if self.hitPointCount % 4 == 0:
                hitPoint = self.find_hit_point(self.arr)
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
           
            bounce = self.find_y_min(self.arr)
            if (bounce is not None) and (np.sum(self.mask[int(bounce[1]), int(bounce[0])]) > 0) \
                    and np.sum(self.bounce == bounce) == 0 and self.isBounce:
                import pdb
                # pdb.set_trace()
                self.bounce.append(bounce)
                if self.new_bounce is not None:
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



