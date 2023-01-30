import os
import sys

### start_stage
# 0: train baseline, create database, train transfer_net and evaluate
# 1: create database, train transfer_net and evaluate
# 2: train transfer_net and evaluate
# 3: evaluate

start_stage = int(sys.argv[1])

#### environment variables
cur_python = "/home/storm/anaconda3/envs/posewarper/bin/python" # '/path/to/your/python/binary'
working_dir = "/home/storm/projects/PoseWarper2/PoseWarper/" # '/path/to/PoseWarper/'

### supplementary files
root_dir = "/home/storm/projects/PoseWarper2/PoseWarper/posewarper_supp_root/PoseWarper_supp_files/" # '/path/to/our/provided/supplementary/files/directory/'

### directory with extracted and renamed frames
img_dir = "/home/storm/PoseWarper/data/posetrack17/renamed_images/"
heatmap_dir = "/home/storm/PoseWarper/data/posetrack17/heatmaps/"

# w32_256x192 config filename
w32_256x192_cfg = "experiments/posetrack/hrnet/w32_256x192_adam_lr1e-4.yaml"
# w48_384x288 config filename
w48_384x288_cfg = "experiments/posetrack/hrnet/w48_384x288_adam_lr1e-4.yaml"
# transfer_net config filename
transfer_net_cfg = "experiments/posetrack/transfer_net.yaml"


### print frequency
PF = 100

#### Paths to other files
json_dir = root_dir + 'posetrack17_json_files/'
w48_384x288_pretrained_coco_model = os.path.join(root_dir, 'pretrained_models/for_baseline_training/pose_hrnet_w48_384x288.pth')
w32_256x192_pretrained_coco_model = os.path.join(root_dir, 'pretrained_models/for_baseline_training/pose_hrnet_w32_256x192.pth')
precomputed_boxes_file = root_dir + 'posetrack17_precomputed_boxes/val_boxes.json'
annot_dir = root_dir + 'posetrack17_annotation_dirs/'

### Output Directories
baseline_w48_384x288_output_dir = os.path.join(root_dir, 'posetrack17_experiments/baseline_w48_384x288/')
if not os.path.exists(baseline_w48_384x288_output_dir):
    os.makedirs(baseline_w48_384x288_output_dir)

baseline_w32_256x192_output_dir = os.path.join(root_dir, 'posetrack17_experiments/baseline_w32_256x192/')
if not os.path.exists(baseline_w32_256x192_output_dir):
    os.makedirs(baseline_w32_256x192_output_dir)

transfer_net_output_dir = os.path.join(root_dir, 'posetrack17_experiments/transfer_net/')
if not os.path.exists(transfer_net_output_dir):
    os.makedirs(transfer_net_output_dir)

heatmap_log_root = os.path.join(root_dir, 'posetrack17_experiments/heatmap_log/')
if not os.path.exists(heatmap_log_root):
    os.makedirs(heatmap_log_root)

light_heatmap_dir = os.path.join(heatmap_dir, "w32_256x192")
if not os.path.exists(light_heatmap_dir):
    os.makedirs(light_heatmap_dir)

large_heatmap_dir = os.path.join(heatmap_dir, "w48_384x288")
if not os.path.exists(large_heatmap_dir):
    os.makedirs(large_heatmap_dir)

target_heatmap_dir = os.path.join(heatmap_dir, "target")
if not os.path.exists(target_heatmap_dir):
    os.makedirs(target_heatmap_dir)


def create_command(script_filename, cfg_path, pretraind_model=None, additionals=[]): #epoch_sfx="", batch_sfx="", rot_factor_sfx="", scale_factor_sfx=""):
  command = " ".join([cur_python, 
    os.path.join(working_dir, "tools", script_filename),
    "--cfg", cfg_path] + additionals + \
    ["OUTPUT_DIR", out_dir,
    "LOG_DIR", log_dir,
    "DATASET.NUM_LABELED_VIDEOS", str(NUM_LABELED_VIDEOS),
    "DATASET.NUM_LABELED_FRAMES_PER_VIDEO", str(NUM_LABELED_FRAMES_PER_VIDEO),
    'DATASET.JSON_DIR', json_dir,
    'DATASET.IMG_DIR', img_dir,
    'MODEL.EVALUATE', "False", 
    "PRINT_FREQ", str(PF)])
  
  if pretraind_model is not None:
        command += " " + 'MODEL.PRETRAINED ' + str(pretraind_model)


  return command


###########
N_str = '115'
V_str = '250'
NUM_LABELED_VIDEOS = -1
NUM_LABELED_FRAMES_PER_VIDEO = -1

if start_stage == 0: # train baseline
  cur_output_dir = os.path.join(baseline_w48_384x288_output_dir, 'V'+V_str+'_N'+N_str + '/')
  if not os.path.exists(cur_output_dir):
      os.makedirs(cur_output_dir)
  
  out_dir = os.path.join(cur_output_dir, 'out/')
  log_dir = os.path.join(cur_output_dir, 'log/')

  epoch_sfx = 'TRAIN.END_EPOCH 20'
  batch_sfx = 'TRAIN.BATCH_SIZE_PER_GPU 8'
  command =  create_command("train.py", w48_384x288_cfg, w48_384x288_pretrained_coco_model,
    [epoch_sfx, batch_sfx])
  print(command)
  os.system(command)

  cur_output_dir = os.path.join(baseline_w32_256x192_output_dir, 'V'+V_str+'_N'+N_str + '/')
  if not os.path.exists(cur_output_dir):
      os.makedirs(cur_output_dir)
  
  out_dir = os.path.join(cur_output_dir, 'out/')
  log_dir = os.path.join(cur_output_dir, 'log/')

  epoch_sfx = 'TRAIN.END_EPOCH 20'
  batch_sfx = 'TRAIN.BATCH_SIZE_PER_GPU 8'
  command =  create_command("train.py", w32_256x192_cfg, pretraind_model=w32_256x192_pretrained_coco_model, 
    additionals = [epoch_sfx, batch_sfx])
  print(command)
  os.system(command)

  exit()
  
if start_stage <= 1: # create database
  cur_output_dir = os.path.join(heatmap_log_root, 'V'+V_str+'_N'+N_str + '/')
  if not os.path.exists(cur_output_dir):
      os.makedirs(cur_output_dir)
  
  out_dir = os.path.join(cur_output_dir, 'out/')
  log_dir = os.path.join(cur_output_dir, 'log/')

  epoch_sfx = ''
  batch_sfx = ''
  rot_factor_sfx = 'DATASET.ROT_FACTOR 0'
  scale_factor_sfx = 'DATASET.SCALE_FACTOR 0.0'
  flip = 'DATASET.FLIP False'
  pretrain_model = os.path.join(baseline_w32_256x192_output_dir, 'V'+V_str+'_N'+N_str + '/', "out", "posetrack/pose_hrnet/w32_256x192_adam_lr1e-4/20_state.pth")
  command =  create_command("create_db.py", w32_256x192_cfg, pretraind_model=pretrain_model, additionals=["--heatmap_root", light_heatmap_dir, epoch_sfx, batch_sfx, rot_factor_sfx, scale_factor_sfx, flip])
  print(command)
  os.system(command)
  pretrain_model = os.path.join(baseline_w48_384x288_output_dir, 'V'+V_str+'_N'+N_str + '/', "out", "posetrack/pose_hrnet/w48_384x288_adam_lr1e-4/20_state.pth")
  command =  create_command("create_db.py", w48_384x288_cfg, pretraind_model=pretrain_model, additionals=["--heatmap_root", large_heatmap_dir, epoch_sfx, batch_sfx, rot_factor_sfx, scale_factor_sfx, flip])
  print(command)
  os.system(command)
 
  exit()

if start_stage <= 2: # train transfer_net
  cur_output_dir = os.path.join(transfer_net_output_dir, 'V'+V_str+'_N'+N_str + '/')
  if not os.path.exists(cur_output_dir):
      os.makedirs(cur_output_dir)
  
  out_dir = os.path.join(cur_output_dir, 'out/')
  log_dir = os.path.join(cur_output_dir, 'log/')

  epoch_sfx = ''
  batch_sfx =  '' # batch_sfx = 'TRAIN.BATCH_SIZE_PER_GPU 8'
  annot_dir = "posetrack17_annotation_dirs/val_withJson/val"

  PF = 100
  command =  create_command("train_transfer.py", transfer_net_cfg, pretraind_model=None, 
    additionals=["--light_heatmap_dir", light_heatmap_dir, "--large_heatmap_dir", large_heatmap_dir, epoch_sfx, batch_sfx,
    "POSETRACK_ANNOT_DIR " + os.path.join(annot_dir, "val_withJson/val"),
    "MODEL.EVALUATE", "False"
    
  ])
  print(command)
  os.system(command)

exit()
if start_stage <= 3: # evaluate
  cur_output_dir = os.path.join(transfer_net_output_dir, 'V'+V_str+'_N'+N_str + '/')
  if not os.path.exists(cur_output_dir):
      os.makedirs(cur_output_dir)
  
  out_dir = os.path.join(cur_output_dir, 'out/')
  log_dir = os.path.join(cur_output_dir, 'log/')

  epoch_sfx = ''
  batch_sfx = ''
  command =  create_command("test.py", transfer_net, out_dir, log_dir, epoch_sfx, batch_sfx)
  print(command)
  os.system(command)