from skimage.transform import ProjectiveTransform
import numpy as np
import pdb
import cv2 as cv
# from PIL import Image, ImageDraw, ImageFont
# 
# import matplotlib.cm 
# from matplotlib.colors import LinearSegmentedColormap 
# import matplotlib.image as mpimg 
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy.random as random
import cv2
import numpy as np
import copy
#------------

recs = {0: ((0, 0), (150, 110)), 1: ((0, 110), (150, 262)), 2: ((0, 262), (150, 414)), 3: ((0, 414), (150, 566)), 4: ((0, 566), (150, 672)),
        5: ((980, 0), (1134, 110)), 6: ((980, 110), (1134, 262)), 7: ((980, 262), (1134, 414)), 8: ((980, 414), (1134, 566)), 9: ((980, 566), (1134, 672))}

def get_t_fun(four_point, height, width, win_h, win_w):
    # height = 457
    # width = 822

    diff_h = (win_h-height)//2
    diff_w = (win_w-width)//2

    t = ProjectiveTransform()
    src = np.asarray(
        [[four_point[0][0], four_point[0][1]], [four_point[1][0], four_point[1][1]], [four_point[2][0], four_point[2][1]], [four_point[3][0], four_point[3][1]]])

    # dst = np.asarray([[300, 100], [300, 100 + height], [300 + width, 100 + height], [300 + width, 100]])
    dst = np.asarray([[diff_w, diff_h], [diff_w, diff_h + height], [diff_w + width, diff_h + height], [diff_w + width, diff_h]])
    if not t.estimate(src, dst): raise Exception("estimate failed")
    return t
def get_hit_point_range(point_range, hit_point, height, width, win_h, win_w, t_fcn):
    hit_point = np.array(hit_point)[:2]
    hit_point = t_fcn(hit_point)

    middle_line = int(win_w / 2)

    first = int((win_h - height)/2)
    second = int(first + height /3)
    third = int(second + height /3)
    forth = int(win_h-(win_h - height)/2)
    
    if hit_point[:,0]<= middle_line:
        if hit_point[:,1] <= first:
            point_range["down"][0] += 1
            return 0 # 'return' was added from Jenny.
        elif hit_point[:,1] <= second:
            point_range["down"][1] += 1
            return 1
        elif hit_point[:,1] <= third:
            point_range["down"][2] += 1
            return 2
        elif hit_point[:,1] <= forth:
            point_range["down"][3] += 1
            return 3
        else:
            point_range["down"][4] += 1
            return 4
    else:
        if hit_point[:,1] <= first:
            point_range["up"][0] += 1
            return 5
        elif hit_point[:,1] <= second:
            point_range["up"][1] += 1
            return 6
        elif hit_point[:,1] <= third:
            point_range["up"][2] += 1
            return 7
        elif hit_point[:,1] <= forth:
            point_range["up"][3] += 1
            return 8
        else:
            point_range["up"][4] += 1
            return 9


def put_text_center(img, text, textY):
    # setup text
    font = cv2.FONT_HERSHEY_SIMPLEX #cv2.FONT_HERSHEY_DUPLEX
    # text = "Hello Joseph!!"

    # get boundary of this text
    textsize = cv2.getTextSize(text, font, 1, 2)[0]

    # get coords based on boundary
    textX = int((img.shape[1] - textsize[0]) / 2)
    # textY = int((img.shape[0] + textsize[1]) / 2)

    # add text centered on image
    cv2.putText(img, text, (textX, textY ), font, 1, (255, 255, 255), 2)
    return img

def get_table_corner_position(height, width, win_h, win_w):
    diff_h = (win_h-height)//2
    diff_w = (win_w-width)//2
    dst = np.asarray([[diff_w, diff_h], [diff_w, diff_h + height], [diff_w + width, diff_h + height], [diff_w + width, diff_h]])
    return dst
def draw_2d_bg(height, width, win_h, win_w, a_name:str='Player A', b_name:str='Player B', name_dis:int=50, black=False):
    # height = 457
    # width = 822
    # win_h = 557
    # win_w = 1000

    img = np.zeros([win_h, win_w, 3],dtype=np.uint8)
    # diff_h = (win_h-height)//2
    # diff_w = (win_w-width)//2
    # dst = np.asarray([[diff_w, diff_h], [diff_w, diff_h + height], [diff_w + width, diff_h + height], [diff_w + width, diff_h]])
    dst = get_table_corner_position(height, width, win_h, win_w)
    if not black:
        # cv.rectangle(img, (dst[0][0], dst[0][1]), (dst[2][0], dst[2][1]), (204, 102, 0), -1)
        cv.rectangle(img, (dst[0][0], dst[0][1]), (dst[2][0], dst[2][1]), (50, 0, 0), -1)
    else:
        cv.rectangle(img, (dst[0][0], dst[0][1]), (dst[2][0], dst[2][1]), (0, 0, 0), -1)

    # cv.rectangle(img, (dst[0][0], dst[0][1]), (dst[2][0], dst[2][1]), (80, 22, 0), -1)
    # import pdb
    # pdb.set_trace()
    # border
    cv.line(img, (dst[0][0], dst[0][1]), (dst[1][0], dst[1][1]), (255, 255, 255), 4)
    cv.line(img, (dst[1][0], dst[1][1]), (dst[2][0], dst[2][1]), (255, 255, 255), 4)
    cv.line(img, (dst[2][0], dst[2][1]), (dst[3][0], dst[3][1]), (255, 255, 255), 4)
    cv.line(img, (dst[3][0], dst[3][1]), (dst[0][0], dst[0][1]), (255, 255, 255), 4)

    # middle line
    cv.line(img, (dst[0][0], int((dst[1][1] + dst[0][1]) / 2)), (dst[2][0], int((dst[2][1] + dst[3][1]) / 2)), (255, 255, 255), 1)
    # net
    cv.line(img, (int((dst[0][0] + dst[3][0]) / 2), dst[0][1]), (int((dst[1][0] + dst[2][0]) / 2), dst[1][1]), (255, 255, 255), 2)
    cv.line(img, (int((dst[0][0] + dst[3][0]) / 2), dst[0][1]), (int((dst[0][0] + dst[3][0]) / 2), dst[0][1] - 15), (255, 255, 255), 3)
    cv.line(img, (int((dst[1][0] + dst[2][0]) / 2), dst[1][1]), (int((dst[1][0] + dst[2][0]) / 2), dst[1][1] + 15), (255, 255, 255), 3)

    x_pos = dst[0][0] + int((dst[3][0] - dst[0][0]) / 2)
    cv.line(img, (x_pos, dst[3][1]), (x_pos, dst[2][1]), (255, 255, 255), 3)

    img = np.rot90(img)
    img = img.copy() # for fix unknow error
    img = put_text_center(img, a_name, name_dis)
    img = put_text_center(img, b_name, win_w-name_dis)
    img = np.rot90(img,3) # rotate 270
    img = img.copy()

    return img

def _2D(t, x,y):
    data_local = t([x, y])
    # cv.circle(result, (int(data_local[0][0]), int(data_local[0][1])), 3, (255, 231, 0), -1)
    # cv.circle(img, (x, y), 3, (255, 255, 255), -1)
    return data_local

def draw_2d(pts_arr, t_fun, img_2d_bg):
    if len(pts_arr) == 0:
        return img_2d_bg, []
    pts_arr = np.array(pts_arr)[:,:2]
    data_local = t_fun(pts_arr)
    for i in range(len(pts_arr)):
        cv.circle(img_2d_bg, (int(data_local[i][0]), int(data_local[i][1])), 10, (255, 231, 0), -1)
    img_2d_bg = np.asarray(img_2d_bg)
    return img_2d_bg, data_local
def frameTo2D(pts_arr, t_fun):
    pts_arr = np.array(pts_arr)[:,:2]
    data_local = t_fun(pts_arr)
    return data_local


def draw_2d_realtime(new_bounce, prev_bounce, hit_point, t_fun, img_2d, hitpt_range_num=-1):
    # if hit_point is not None:
       
    #     hit_point = np.array(hit_point)[:2]
    #     hit_point = t_fun(hit_point)

    #     cv.circle(img_2d, (int(hit_point[:,0]), int(hit_point[:,1])), 10, (255, 231, 255), -1)
    
    if prev_bounce is not None:
        prev_bounce = np.array(prev_bounce)[:2]
        prev_bounce = t_fun(prev_bounce)

        cv.circle(img_2d, (int(prev_bounce[:,0]), int(prev_bounce[:,1])), 10, (255, 231, 0), -1)
    
    if new_bounce is not None:
        new_bounce = np.array(new_bounce)[:2]
        new_bounce = t_fun(new_bounce)
        cv.circle(img_2d, (int(new_bounce[:,0]), int(new_bounce[:,1])), 10, (0, 0, 255), -1) # high light
    
    if hitpt_range_num != -1:
        cv2.rectangle(img_2d, (0, 0), (150, 672), (0, 0, 0), -1)
        cv2.rectangle(img_2d, (980, 0), (1134, 672), (0, 0, 0), -1)
        loc = recs[hitpt_range_num]
        cv2.rectangle(img_2d, loc[0], loc[1], (100, 100, 100), -1)
        
    return img_2d

# def paint_chinese_opencv(im, chinese, pos, color):
#     img_PIL = Image.fromarray(cv.cvtColor(im, cv.COLOR_BGR2RGB))
#     font = ImageFont.truetype('NotoSansCJK-Bold.ttc', 25)
#     fillColor = color
#     position = pos
#     draw = ImageDraw.Draw(img_PIL)
#     draw.text(position, chinese, font=font, fill=fillColor, anchor='mm', align='center')
#  
#     img = cv.cvtColor(np.asarray(img_PIL),cv.COLOR_RGB2BGR)
#     return img
# 
# def probability_map(points, n, height, width, win_h, win_w):
#   prob_map = np.zeros((n, 4))
# 
#   diff_h = (win_h-height)//2
#   diff_w = (win_w-width)//2
#   # top = diff_h
#   # left = diff_w
# 
#   prob_map[0, 0] += len(points[(points[:, 0] > diff_w) & (points[:, 0] < diff_w + width//4) & (points[:, 1] > diff_h) & (points[:, 1] < diff_h + height//n)])
#   prob_map[0, 1] += len(points[(points[:, 0] > diff_w + width//4) & (points[:, 0] < win_w//2) & (points[:, 1] > diff_h) & (points[:, 1] < diff_h + height//n)])
#   prob_map[0, 2] += len(points[(points[:, 0] > win_w//2) & (points[:, 0] < win_w//2 + width//4) & (points[:, 1] > diff_h) & (points[:, 1] < diff_h + height//n)])
#   prob_map[0, 3] += len(points[(points[:, 0] > win_w//2 + width//4) & (points[:, 1] > diff_h) & (points[:, 1] < diff_h + height//n)])
# 
#   prob_map[1, 0] += len(points[(points[:, 0] > diff_w) & (points[:, 0] < diff_w + width//4) & (points[:, 1] > diff_h + height//n) & (points[:, 1] < diff_h + 2*height//n)])
#   prob_map[1, 1] += len(points[(points[:, 0] > diff_w + width//4) & (points[:, 0] < win_w//2) & (points[:, 1] > diff_h + height//n) & (points[:, 1] < diff_h + 2*height//n)])
#   prob_map[1, 2] += len(points[(points[:, 0] > win_w//2) & (points[:, 0] < win_w//2 + width//4) & (points[:, 1] > diff_h + height//n) & (points[:, 1] < diff_h + 2*height//n)])
#   prob_map[1, 3] += len(points[(points[:, 0] > win_w//2 + width//4) & (points[:, 1] > diff_h + height//n) & (points[:, 1] < diff_h + 2*height//n)])
# 
#   if n == 3:
#       prob_map[2, 0] += len(points[(points[:, 0] > diff_w) & (points[:, 0] < diff_w + width//4) & (points[:, 1] > diff_h + 2*height//n)])
#       prob_map[2, 1] += len(points[(points[:, 0] > diff_w + width//4) & (points[:, 0] < win_w//2) & (points[:, 1] > diff_h + 2*height//n)])
#       prob_map[2, 2] += len(points[(points[:, 0] > win_w//2) & (points[:, 0] < win_w//2 + width//4) & (points[:, 1] > diff_h + 2*height//n)])
#       prob_map[2, 3] += len(points[(points[:, 0] > win_w//2 + width//4) & (points[:, 1] > diff_h + 2*height//n)])
# 
#   # print(points)
#   # print(points[(points[:, 0] > width//2) & (points[:, 0] < width//2 + width//4) & (points[:, 1] > diff_h + height//n) & (points[:, 1] < diff_h + 2*height//n)])
#   # print(diff_w + width//4, width//2)
#   # print(width//2, width//2 + width//4)
#   # print(prob_map)
# 
#   prob_map[:, :2] = prob_map[:, :2]/np.sum(prob_map[:, :2])
#   prob_map[:, 2:] = prob_map[:, 2:]/np.sum(prob_map[:, 2:])
# 
#   # print(prob_map)
#   return prob_map
# 
# def make_Ramp( ramp_colors ):
#     from colour import Color
#     color_ramp = LinearSegmentedColormap.from_list('my_list', [Color(c1).rgb for c1 in ramp_colors])
#     plt.figure(figsize=(15, 3))
#     plt.imshow([list(np.arange(0, len( ramp_colors ), 0.1)) ], interpolation='nearest', origin='lower', cmap= color_ramp)
#     plt.xticks([])
#     plt.yticks([])
#     return color_ramp
# 
# def get_heatmap(prob_map, table_h, table_w, win_h, win_w):
#   custom_ramp = make_Ramp(['#AAEBF8','#4FA8D7','#3695D3','#0C5BCD'])
# 
#   sns.set()
#   fig, ax = plt.subplots(figsize=(table_h/100, table_w/100), dpi=600)
#   # ax = sns.heatmap(prob_map, cmap=matplotlib.cm.winter, alpha=0.5, annot=True, zorder=2, cbar=False, fmt ='.0%', annot_kws={'size':20, 'fontweight': 'bold'}, linewidths=1.5)
#   ax = sns.heatmap(prob_map, cmap=custom_ramp, alpha=0.5, annot=True, zorder=2, cbar=False, fmt ='.0%', annot_kws={'size':20, 'fontweight': 'bold'}, linewidths=1.5)
#   # plt.imshow(img_2d, aspect=ax.get_aspect(), extent=ax.get_xlim()+ax.get_ylim(), zorder=1) #put the map under the heatmap
# 
#   plt.axis('off')
#   # plt.savefig('probability_map.png', bbox_inches='tight', pad_inches=0)
#   # plt.show()
# 
#   plt.tight_layout()
# 
#   canvas = plt.gca().figure.canvas
#   canvas.draw()
#   data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
#   prob_map_img = data.reshape(canvas.get_width_height()[::-1] + (3,))
#   prob_map_img = cv.cvtColor(prob_map_img, cv.COLOR_RGB2BGR)
# 
#   return prob_map_img
# 
# def add_img_hm(img_2d, prob_map_img, table_h, table_w, win_h, win_w):
#   # img_2d = img_2d.copy()
#   # cv.rectangle(img_2d, ((win_h-table_h)//2, (win_w-table_w)//2), ((win_h-table_h)//2+table_h, (win_w-table_w)//2+table_w), (0, 0, 0), -1)
#   # cv.rectangle(img_2d, (20, 30), (50, 60), (255, 255, 255), -1)
# 
#   pad_w = (win_w-table_w)//2
#   pad_h = (win_h-table_h)//2
#   prob_map_img = cv.resize(prob_map_img, (table_h, table_w))
#   print(prob_map_img.shape)
#   prob_map_img[:table_h-pad_h//2+10, :win_w//2] = (cv.cvtColor(prob_map_img[:table_h-pad_h//2+10, :win_w//2], cv.COLOR_RGB2BGR)*0.5).astype(np.uint8)
# 
#   prob_map_img = cv.copyMakeBorder(prob_map_img, pad_w, pad_w, pad_h, pad_h, cv.BORDER_CONSTANT, value=(0, 0, 0))
#   result = cv.addWeighted(img_2d, 0.3, prob_map_img, 1.0, 0.0)
#   # result = paint_chinese_opencv(result, '成功大學 蔡宗翰', (win_h//2-80, 20), (255, 255, 255))
#   # result = paint_chinese_opencv(result, '成功大學 李小英', (win_h//2-80, win_w-60), (255, 255, 255))
#   result = cv.line(result, (pad_h-15, win_w//2), (pad_h+table_h+15, win_w//2), (0, 0, 0), 5)
#   return result

def merge_image(src1, src2, src3, src3_low, table_h, table_w, win_h, win_w):
    # resize = cv.resize(src1, (1400, 990))
    # result = cv.resize(src2, (520, 990))

    # resize = cv.resize(src1, (1632, 918)) #*0.75
    # result = cv.resize(src2, (288, 918))
    resize = cv.resize(src1, (1440, 810)) #*0.75
    result = cv.resize(src2, (480, 810))
    # print(resize.shape)
    # print(result.shape)
    # import pdb
    # pdb.set_trace()

    hstack = np.hstack((resize, result))
    vstack = np.vstack((hstack, src3))
    vstack = np.vstack((src3_low, vstack))

    return vstack
def color_mask(img, x0, x1, y0, y1, _color, text:str=''):
    img_mask = copy.deepcopy(img)
    # cv2.rectangle(img_mask, (x0, y0), (x1, y1), (0, 255, 0), -1)
    # cv2.rectangle(img_mask, (x0, y0), (x1, y1), _color, -1)
    # img_mask[x0:x1,y0:y1] = img_mask[x0:x1,y0:y1] + np.array(_color, dtype=np.uint8)
    img_mask[y0:y1,x0:x1] = img_mask[y0:y1,x0:x1] + np.array(_color, dtype=np.uint8)
    
    # alpha 为第一张图片的透明度
    alpha = 0
    # beta 为第二张图片的透明度
    beta = 1
    gamma = 0
    # cv2.addWeighted 将原始图片与 mask 融合
    img = img_mask
    # img = cv2.addWeighted(img, alpha, img_mask, beta, gamma)

    

    # cv2.putText(img, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 
    #                         2, (255, 255, 255), 2, cv2.LINE_AA)
    return img

def lower_uper(arr):
    return int(np.min(arr)), int(np.max(arr))


# def get_heatmap(img_2d, bounce_2d, x0, x1, y0, y1, color_rg ,x_patch=2, y_patch=2):
def get_heatmap(img_2d, bounce_2d, x0, x1, y0, y1, color_rg ,x_patch=2, y_patch=3, returnImg=True):

    def patch_counter(x0:int, x1:int, y0:int, y1:int, arr) ->  int:
        count = np.sum((x0<arr[:,0])*(arr[:,0]<x1)*(y0<arr[:,1])*(arr[:,1]<y1))
        return count

    #          table
    # (x1,y0)---------
    #        |patch A|
    # (cx,y0) ---------
    #        |patch B|
    # (x0,y0)---------

    # patch B
    patchB_count = patch_counter(x0, x1, y0, y1, bounce_2d)


    
    # (cx,y0)---------
    #         |p2, p3|
    #         |p0, p1|
    # (x0,y0)---------(x0,y1)

    # *** notice cx, y1
    count_result = []
    x_y_count = []
    delta_x = int(((x1-x0)/x_patch) + 0.5) # +0.5 is for round
    delta_y = int(((y1-y0)/y_patch) + 0.5)

    for i in range(x_patch):
        px0 = x0 + (i*delta_x)
        px1 = x0 + ((i+1)*delta_x) # +1, because range is start from 0
        for j in range(y_patch):
            py0 = y0 + (j*delta_y)
            py1 = y0 + ((j+1)*delta_y)
            
            # count_result.append()
            count = patch_counter(px0, px1, py0, py1, bounce_2d)
            # pdb.set_trace()
            p = (count/patchB_count) if patchB_count > 0 else 0
            if count != 0 and color_rg=='g':
                _color = (0, int(180*p), 0)  # bgr
                _color = (np.array(_color) + np.array([0, 75, 50])).tolist()
            elif count != 0 and color_rg=='r':
                _color = (0, 0, int(180*p))  # bgr
                _color = (np.array(_color) + np.array([0, 50, 75])).tolist()
            else:
                _color = (0,0,0)
            if returnImg:
                img_2d = color_mask(img_2d, px0, px1, py0, py1, _color, \
                                "{}".format(str(count)))
            
            x_y_count.append([px0, py0, count])

    
    return img_2d, x_y_count
def put_hm_text(img_2d_90, x_y_count):
    #  -->x
    # |
    # V
    # y

    f = cv2.FONT_HERSHEY_SIMPLEX #cv2.FONT_HERSHEY_DUPLEX
    x_y_count = np.array(x_y_count)
    
    if np.sum(x_y_count[:,2]) != 0:
        x_y_count[:,2] = ((x_y_count[:,2]/np.sum(x_y_count[:,2]))*100).astype(int)

    w, h, c = img_2d_90.shape #11xx, 
    img_2d_90 = img_2d_90.copy() # for fix some error
    for x,y,count in x_y_count:
        cv2.putText(img_2d_90, str(count)+'%', (y+10, w-x-10), f, 
                            1, (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.putText(img_2d_90, str(y)+str(x), (y, w-x+10), cv2.FONT_HERSHEY_SIMPLEX, 
        #                     2, (255, 255, 255), 2, cv2.LINE_AA)
    return img_2d_90

def put_hm_text_h(img_2d, x_y_count):
    #  -->x
    # |
    # V
    # y

    f = cv2.FONT_HERSHEY_SIMPLEX #cv2.FONT_HERSHEY_DUPLEX

    x_y_count = np.array(x_y_count)
    if np.sum(x_y_count[:,2]) != 0:
        x_y_count[:,2] = ((x_y_count[:,2]/np.sum(x_y_count[:,2]))*100).astype(int)

    # w, h, c = img_2d_90.shape #11xx, 
    # img_2d = (np.rot90(img_2d_90, 3)).copy() # for fix some error
    for x,y,count in x_y_count:
        cv2.putText(img_2d, str(count)+'%', (x+10, y+30), f, 
                            1, (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.putText(img_2d_90, str(y)+str(x), (y, w-x+10), cv2.FONT_HERSHEY_SIMPLEX, 
        #                     2, (255, 255, 255), 2, cv2.LINE_AA)
    return img_2d

def get_bg_mask(file_path):
	import pdb
	# pdb.set_trace()
	if file_path is not None:
		bg_mask = cv2.imread(file_path)
		tmp = bg_mask.copy()
		bg_mask[tmp==255]=0
		bg_mask[tmp!=255]=1
	else:
		bg_mask = None
	# TODO
	# pdb.set_trace()
	return bg_mask
if __name__ == '__main__':
    # table_h = 456
    # table_w = 822
    # win_h = 556
    # win_w = 1000
    # ndim = 2

    table_h = 456
    table_w = 822
    win_h = 672#556
    win_w = 1134#1022
    ndim = 2

    pad_w = (win_w-table_w)//2
    pad_h = (win_h-table_h)//2

    ref_pts = np.load('C:/Users/OWNER/Documents/Jenny/20210323/points_arr.npy')
    t_fun = get_t_fun(ref_pts, table_h, table_w, win_h, win_w)

    img_2d_bg = draw_2d_bg(table_h, table_w, win_h, win_w)
    img_2d_bg[pad_h:pad_h+table_h, win_w//2:win_w-pad_w] = (0, 30, 255)

    bounce = [[1300, 800], [1200, 700], [500, 800], [700, 800]]
    img_2d, trans_points = draw_2d(bounce, t_fun, img_2d_bg)
    cv.line(img_2d, (win_w//2, pad_h), (win_w//2, pad_h+table_h), (255, 255, 0), 10)
    # cv.imshow('img_2d',img_2d)
    # cv.waitKey(0)
    prob_map = probability_map(trans_points, ndim, table_h, table_w, win_h, win_w)
    prob_map = np.rot90(prob_map)
    img_2d = np.rot90(img_2d)
    cv.imshow('img_2d',img_2d)
    cv.waitKey(0)

    prob_map_img = get_heatmap(prob_map, table_h, table_w, win_h, win_w)
    # prob_map_img = prob_map_img[100:prob_map.shape[0]-100, 100:prob_map.shape[1]-100]
    prob_map_img = prob_map_img[100:prob_map.shape[0]-300, 280:prob_map.shape[1]-100]
    result = add_img_hm(img_2d, prob_map_img, table_h, table_w, win_h, win_w)

    # cv.imshow("plot", prob_map_img)
    cv.imshow("plot", result)
    cv.waitKey(0)


