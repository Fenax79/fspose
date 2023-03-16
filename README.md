# FSPose: A Heterogeneous Framework with Fast and Slow Networks for Human Pose Estimation in Videos (to appear at IEICE Transactions on Information and Systems, Vol.E106-D, No.6, Jun.2023 )

## License
FSPose is released under the Apache 2.0 license.
## Prepare Data
Download posetrack data.
### Data Directory Tree
Data  
└posetrack18  
&nbsp;&nbsp;&nbsp;&nbsp;├annotations  
&nbsp;&nbsp;&nbsp;&nbsp;└images  

<!--
Posetack17's images are needed to rename to renamed_images with [this script](https://github.com/facebookresearch/DetectAndTrack/blob/master/tools/gen_posetrack_json.py).
-->
## Install
Modify DATA path and MODEL_FOLDER path in docker-compose.yml.

docker-compose(17.12.0+)
```
# install docker-compose
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

Download supp files from [this link](https://www.dropbox.com/s/ygfy6r8nitoggfq/PoseWarper_supp_files.zip?dl=0) and unzip to codes/PoseWarper_supp_files
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
docker-compose build && docker-compose up -d
docker-compose exec posewarper bash
cd lib && make && cd deform_conv && python setup.py develop
```

## Prepare Models
Put the model data in the ./models folder to your model folder.\
Note that you have to decompress the compressed model in ./models/posetrack18/pose_hrnet/w32_256x192_adam_lr1e-4.\
We have compressed the model into two zip files because of the size limitation of github.
### Model Directory Tree
YourModelFolder  
├posetrack18  
&nbsp;&nbsp;&nbsp;&nbsp;├pose_hrnet  
&nbsp;&nbsp;&nbsp;&nbsp;│└w32_256x192_adam_lr1e-4/final_state.pth  
&nbsp;&nbsp;&nbsp;&nbsp;├mobilenet_test_1/100epoch.pth  
&nbsp;&nbsp;&nbsp;&nbsp;├transfer_net_56x56_mobilenetx2/20epoch.pth  
&nbsp;&nbsp;&nbsp;&nbsp;└transfer_net_64x48_w32_mobilenet/20epoch.pth  

## Create Heatmap
In Container, run it.

```
# for light(w32_256x192)
python tools/create_db.py \
    --cfg experiments/posetrack/hrnet/w32_256x192_adam_lr1e-4.yaml \
    --heatmap_root /root/data/posetrack18/heatmaps/w32_256x192 \
    OUTPUT_DIR ${SUPP_FILES}/posetrack18_experiments/heatmap_log/V250_N115/out/ \
    LOG_DIR ${SUPP_FILES}/posetrack18_experiments/heatmap_log/V250_N115/log/ \
    DATASET.NUM_LABELED_VIDEOS -1 \
    DATASET.NUM_LABELED_FRAMES_PER_VIDEO -1 \
    DATASET.JSON_DIR ${SUPP_FILES}/posetrack18_json_files/ \
    DATASET.IMG_DIR /root/data/posetrack18/ \
    MODEL.PRETRAINED /root/model/posetrack18/pose_hrnet/w32_256x192_adam_lr1e-4/final_state.pth \
    MODEL.EVALUATE False \
    PRINT_FREQ 100 \
    DATASET.ROT_FACTOR 0 \
    DATASET.SCALE_FACTOR 0.0 \
    DATASET.FLIP False \
    TRAIN.BATCH_SIZE_PER_GPU 8

# for mobilenet
python tools/create_db.py \
    --cfg experiments/posetrack/mobilenet/test_1.yaml \
    --heatmap_root /root/data/posetrack18/heatmaps/mobilenet_test_1 \
    --mobilenet \
    OUTPUT_DIR ${SUPP_FILES}/posetrack18_experiments/heatmap_log/V250_N115/out/ \
    LOG_DIR ${SUPP_FILES}/posetrack18_experiments/heatmap_log/V250_N115/log/ \
    DATASET.NUM_LABELED_VIDEOS -1 \
    DATASET.NUM_LABELED_FRAMES_PER_VIDEO -1 \
    DATASET.JSON_DIR ${SUPP_FILES}/posetrack18_json_files/ \
    DATASET.IMG_DIR /root/data/posetrack18/ \
    POSETRACK_ANNOT_DIR ${SUPP_FILES}/posetrack18_annotation_dirs/val/ \
    DATASET.IS_POSETRACK18 True \
    MODEL.PRETRAINED /root/model/posetrack18/mobilenet/test_1/100epoch.pth \
    MODEL.EVALUATE False \
    PRINT_FREQ 100 \
    DATASET.ROT_FACTOR 0 \
    DATASET.SCALE_FACTOR 0.0 \
    DATASET.FLIP False \
    TRAIN.BATCH_SIZE_PER_GPU 8
```

## Prediction example
In Container, run it.

```
# valdiate_type:0 2heatmap, 2 3heatnap, 4: 2.5heatmap 2model,  5: merged_net

python tools/validate_new_methods.py \
    --cfg_lightx2 experiments/posetrack/transfer_net_56x56.yaml \
    --lightx2_model /root/model/posetrack18/transfer_net_56x56_mobilenetx2/20epoch.pth \
    --cfg experiments/posetrack/transfer_net_64x48.yaml \
    --validate_type 0 \
    --light_heatmap_dir /root/data/posetrack18/heatmaps/mobilenet_test_1 \
    --light_origin_image_width 224 \
    --light_origin_image_height 224 \
    --large_heatmap_dir /root/data/posetrack18/heatmaps/w32_256x192 \
    --large_origin_image_width 192 \
    --large_origin_image_height 256 \
    --three_hp \
    --i_frame_intreval 16 \
    OUTPUT_DIR ${SUPP_FILES}/posetrack18_experiments/transfer_net/V250_N115/out/ \
    LOG_DIR ${SUPP_FILES}/posetrack18_experiments/transfer_net/V250_N115/log/ \
    POSETRACK_ANNOT_DIR ${SUPP_FILES}/posetrack18_annotation_dirs/val \
    DATASET.NUM_LABELED_VIDEOS -1 \
    DATASET.NUM_LABELED_FRAMES_PER_VIDEO -1 \
    DATASET.JSON_DIR ${SUPP_FILES}/posetrack18_json_files/ \
    DATASET.IMG_DIR /root/data/posetrack18/ \
    MODEL.PRETRAINED /root/model/posetrack18/transfer_net_64x48_w32_mobilenet/20epoch.pth \
    MODEL.EVALUATE False \
    PRINT_FREQ 100 \
    DATASET.IS_POSETRACK18 True
```

## Acknowledgement

Our FSPose implementation is built on top of [*PoseWarper*](https://github.com/facebookresearch/PoseWarper) and [*MobilePose*](https://github.com/YuliangXiu/MobilePose). We thank the authors for releasing their code.

