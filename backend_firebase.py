import os
import time
import json
import numpy as np

from fun_2d_ne import *

name1 = 'NTCU'
name2 = 'NKNU'

def report(cap, report_hand_hit_queue, report_bounce_queue):
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

def upload_report(cap):
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
        hit_region = None
        with open('hit_region.json', 'r') as f:
            hit_region = json.load(f)
        
        for key in hit_region.keys():
            for k in hit_region[key].keys():
                hit_region[key][k] = [hit_region[key][k]]
            hit_region[key] = [hit_region[key]]
        hit_region = [hit_region]

        bounce_region = {"left": {"pos_neg": ["0", "0", "0", "0", "0"], "positive": ["0", "0", "0", "0", "0"], "negtive": ["0", "0", "0", "0", "0"], "ratio": ["0", "0"]}, "right": {"pos_neg": ["0", "0", "0", "0", "0"], "positive": ["0", "0", "0", "0", "0"], "negtive": ["0", "0", "0", "0", "0"], "ratio": ["0", "0"]}}
        try:
            with open('bounce_region.json', 'r') as f:
                bounce_region = json.load(f)
        except:
            print('Cannot load bounce_region.json. Use default.')
        bounce_region = [bounce_region]

        region = [{'hit_region': hit_region, 'bounce_region': bounce_region}]

        old = None
        if os.path.isfile('testing2.json'):
            with open('testing2.json', 'r') as f:
                old = json.load(f)
            idx = len(old['time'])
            old['time'].append({str(idx): region})  # append results            
        
        else:
            idx = 0
            old = {'time': []}
            old['time'].append({str(idx): region})  # append results
        
        with open('testing2.json', 'w') as f:
            json.dump(old, f)

        json_name = 'testing2.json'
        storage.child(json_name).put(json_name)

        time.sleep(1)