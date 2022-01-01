import pyopenpose as op

def pose_estimate(op, opwrapper, pose_queue, kp_queue, skeleton_mask):
    first_print = False
    while cap.isOpened():
        if pose_queue.empty():
            continue
        start = time.time()
        idx, frame = pose_queue.get()
        # print('deq', time.time()-start)
        if frame is not None:
            start = time.time()
            datum = op.Datum()
            masked_img = apply_mask(frame, skeleton_mask)

            if first_print:
                cv2.imwrite('test_mask.png', masked_img)
                first_print = False
            
            datum.cvInputData = masked_img
            opwrapper.emplaceAndPop(op.VectorDatum([datum]))
            if datum.poseKeypoints is None:
                continue
            else:
                keypoints = datum.poseKeypoints[:2, 2:8, :-1]
            
            kp_queue.put((idx, keypoints))
            if kp_queue.full():
                kp_queue.get()
            
    
def wait_for_hit_point(kp_queue, hitpoint_queue, hand_queue, left_handed):
    while cap.isOpened():
        if hitpoint_queue.empty():
            continue
        else:
            hit_point = hitpoint_queue.get()
            is_forehand = judge_fore_back(kp_queue, hit_point, left_handed)
            is_forehand = 'forehand' if is_forehand else 'backhand'
            hand_queue.put(is_forehand)

def fore_or_back(angles):
    is_forehand = True
    diff = angles[-1] - angles[0]
    if diff >= 0:
        is_forehand = False    
    else:
        is_forehand = True

    return is_forehand

def judge_fore_back(kp_queue, hit_point, left_handed=(False, False)):
    N = 30
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
    if count > N:
        keypoints = keypoints[-N:]

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
    
    angles = []
    frame_idx = []
    for idx, arm in arm_kps:
        angle = calculate_angle(arm[0], arm[1], arm[2])
        angles.append(angle)
        frame_idx.append(idx)
    
    if np.isnan(angles).any():
        s = pd.Series(angles)
        s = s.interpolate()
        angles = s.tolist()
    
    angles = moving_average(angles, 10)
    is_forhand = fore_or_back(angles)
    
    return is_forhand