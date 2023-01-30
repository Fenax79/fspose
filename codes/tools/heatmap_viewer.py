import json
import cv2
import os
import torch
import glob
import numpy as np

heatmap_root = "/home/storm/projects/PoseWarper2/PoseWarper/posewarper_supp_root/PoseWarper_supp_files/posetrack17_experiments/heatmap_log/V250_N115/out/posetrack/pose_hrnet/w32_256x192_adam_lr1e-4"
#video_title = "bonn_mpii_train_5sec/01686_mpii"
video_title = "bonn_5sec/015860_mpii"
# target_track_id = 1
target_track_id = 4
viode_outputname = '20210209_output_valid.avi'
width = 48
height = 64

frame_root_folder = os.path.join(heatmap_root, video_title)
each_frame_folders = os.listdir(frame_root_folder)

each_frame_folders = list(map(lambda x: os.path.join(frame_root_folder, x), each_frame_folders))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(viode_outputname,fourcc, 20.0, (width,height))
target_track_id_filename = str(target_track_id) + ".pt"

base = np.zeros((height, width, 3), dtype=np.uint8)

start_folder_index = -1
# search first frame.
for i, frame_folder in enumerate(each_frame_folders):
    if os.path.exists(os.path.join(frame_folder, target_track_id_filename)):
        start_folder_index = i
        break

assert start_folder_index != -1

for i in range(start_folder_index, len(each_frame_folders)):
    frame_folder = each_frame_folders[i]

    if os.path.exists(os.path.join(frame_folder, target_track_id_filename)):
        output = torch.load(os.path.join(frame_folder, target_track_id_filename))
        num_0_heatmap = output[0].to('cpu').detach().numpy().copy()
        total = np.zeros(num_0_heatmap.shape, dtype=num_0_heatmap.dtype)

        for j in range(output.shape[0]):
            img = output[j].to('cpu').detach().numpy().copy()
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            total += output[j].to('cpu').detach().numpy().copy()
            # cv2.imshow("img", img)
            # cv2.waitKey(0)
        
        total = np.clip(total * 255, 0, 255).astype(np.uint8)
        total = cv2.cvtColor(total, cv2.COLOR_GRAY2RGB)
        # cv2.imshow("total", total)
        # cv2.waitKey(0)
    
        out.write(total)
    else:
        out.write(base)

out.release()



'''
while video == train_json["images"][image_index]["file_name"].split("/")[-2]:
    filepath = os.path.join(image_root, train_json["images"][image_index]["file_name"])
    image = cv2.imread(filepath)
    image_id = train_json["images"][image_index]["id"]

    ann = train_json["annotations"][ann_index]
    while image_index + 1 == ann["image_id"]:

        if target_track_id == ann["track_id"]:
            bbox = ann["bbox"]
            image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0, 0, 255), 3)

        ann_index += 1
        ann = train_json["annotations"][ann_index]

    out.write(image)
    image_index += 1

out.release()
'''