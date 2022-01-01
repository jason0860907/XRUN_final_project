import numpy as np
import cv2
import random
import time

from trace_record import *

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

def drawing(cap, hitPoint_queue, frame_queue, detections_queue, fps_queue, mask, frame_rgb_queue, draw_original, 
    time_queue, t_fun, segRally_thres, image_queue, tail_queue, report_hand_hit_queue, report_bounce_queue):
    noTail_count = 0    
    import time
    random.seed(3)  # deterministic bbox colors
    tail = TraceRecord(6, mask)
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
                tail.inQ(hitPoint_queue, bboxes, report_bounce_queue, report_hand_hit_queue)
            else:
                tail.inQ(hitPoint_queue, None, report_bounce_queue, report_hand_hit_queue)
            
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

def showing(cap, width, height, show_2d, dont_show, out_filename, image_queue, hand_queue, mask, tail_queue, 
	t_fun, draw_original, name1, name2, watermark_img, frame_queue_toShowing, 
	report_hand_hit_queue, hitPoint_queue):
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
            # if not hitPoint_queue.empty():
            #     hitPoint_queue.get()
                # import pdb
                
                # print('612 report_bounce_queue',report_bounce_queue.qsize(),' report_hand_hit_queue',report_hand_hit_queue.qsize())
                # pdb.set_trace()
            if not hand_queue.empty():
                hand = hand_queue.get()
           #    # jenny
                if hand =='forehand':
                    hand_num = 0
                else:
                    hand_num = 1
                # hand_num = 2 # no hand info.
                # print('--before report_hand_hit_queue put\n')
                report_hand_hit_queue.put([hand_num, tail.hitPoint_range_num])
                # print('--after report_hand_hit_queue put\n')

            # ---------------
            # pdb.set_trace()
            # if out_filename is not None:
            #     video.write(image)

            if show_2d:
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

            if not dont_show:
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

                cv2.putText(show_img, 'Detect: {}'.format(hand), (800, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
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