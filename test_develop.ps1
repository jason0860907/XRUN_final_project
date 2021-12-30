# set -e


$darknet_path = 'C:\Users\OWNER\Documents\xrun\darknet'
$exp_pwd = $((pwd).Path)
echo ${exp_pwd}

conda activate pylon

# --out_filename ${exp_pwd}/videos/ntut2.mp4
# --input ${exp_pwd}/videos/C0043.MP4 `
# python .\tabletennis_weichi_test.py `
# python .\tabletennis_pose_211014_v0_1_jenny.py `
python .\tabletennis_pose_211014_v0_4_jenny.py `
    --input ${exp_pwd}/videos/C0043.MP4 `
    --weights ${darknet_path}/weights/211012/yolo-obj_3000.weights `
    --config_file ${darknet_path}/cfg/yolo-obj.cfg `
    --data_file ${darknet_path}/data/table-tennis.data `
    --bounce_mask ${exp_pwd}/imgs/mask.jpg `
    --draw_original `
    --show_2d `
    --ref_pts ${exp_pwd}/points_arr.npy `
    --bg_mask ${exp_pwd}/imgs/bg_mask.jpg `
    --fps_skip 4 `
    --out_filename ${exp_pwd}/videos/testing2.MP4