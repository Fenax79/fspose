import json
import cv2
import os

# train_json_file = "/home/storm/projects/PoseWarper2/PoseWarper/posewarper_supp_root/PoseWarper_supp_files/posetrack17_json_files/posetrack_train.json"
train_json_file = "/home/storm/projects/PoseWarper2/PoseWarper/posewarper_supp_root/PoseWarper_supp_files/posetrack17_json_files/posetrack_val.json"
image_root = "/home/storm/PoseWarper/data/posetrack17/renamed_images"

with open(train_json_file, "r") as f:
    train_json = json.load(f)


ann_index = 0
image_index = 0
video = train_json["images"][image_index]["file_name"].split("/")[-2]
width = int(train_json["images"][image_index]["width"])
height = int(train_json["images"][image_index]["height"])
target_track_id = 4 # 1

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_train.avi',fourcc, 20.0, (width,height))

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
print("hoge")

for i, ann in enumerate(train_json["annotations"]):
    if ann["track_id"] == target_track_id:
        if video == train_json["images"][ann["image_id"] - 1]["file_name"].split("/")[-2]:
            print(i, ann["image_id"])
