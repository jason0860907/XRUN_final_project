import cv2
import numpy as np
from numpy import ndarray
from typing import List, Tuple

BLACK = (255, 255, 255)

def define_mask_area() -> List[ndarray]:
    areas = []    
    # up_points = np.array([[0, 0], [1920, 0], [1920, 210], [0, 460]])
    # person_1 = np.array([[730, 350], [730, 540], [575, 540], [575, 350]])
    # person_2 = np.array([[[1920, 210], [1730, 210], [1730, 550], [1580, 540], [1580, 1080], [1920, 1080]]])
    # areas += [up_points, person_1, person_2]
    area_1 = np.array([[0, 0], [1920, 0], [1920, 460], [0, 460]])
    areas += [area_1]
    return areas

def fill_mask_area(mask: ndarray, areas: List[ndarray], color: Tuple) -> None:
    for area in areas:
        cv2.fillPoly(mask, [area], color)
    
def generate_mask(mask_size: Tuple = (1080, 1920), color: Tuple = BLACK):
    # mask size: (H, W)
    mask = np.ones(mask_size, np.uint8)
    
    # define mask areas
    mask_areas = define_mask_area()

    # fill the areas with color
    fill_mask_area(mask, mask_areas, color)

    # reverse
    mask = cv2.bitwise_not(mask)

    return mask

def apply_mask(img, mask):
    # print(img.shape, mask.shape, img.dtype, mask.dtype)
    return cv2.bitwise_and(img, img, mask=mask)
    # return img * mask

# mask = generate_mask((1080, 1920), BLACK)

# cv2.imwrite('skeleton_mask.png', mask, )

# if __name__ == '__main__':
#     cap = cv2.VideoCapture(0)
#     while cap.isOpened():
#         ret, frame = cap.read()

#         cv2.imwrite('cap.png', frame)
#         break
#     cap.release()
#     # mask = generate_mask()
#     # cv2.imshow('mask', mask)
#     # cv2.waitKey(10000)
#     # cv2.destroyAllWindows()