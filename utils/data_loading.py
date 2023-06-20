import cv2
import json
import numpy as np

# ==================================================================================================================

def get_depth(disparity, center, w, h, ratio):

    bbox = np.array([center[0] - w / 2, center[1] - h / 2, center[0] + w / 2, center[1] + h / 2]).astype(int)
    crop = disparity[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2]]
    crop_center = np.array([crop.shape[0] / 2, crop.shape[1] / 2]).astype(int)

    if crop[crop_center[0]][crop_center[1]] < 220:
      depth = crop[crop_center[0]][crop_center[1]] / ratio
    else:
      # if no depth in center, calculate average depth
      crop_count = 0
      crop_sum = 0
      for x in range(crop.shape[0]):
        for y in range(crop.shape[1]):
          if crop[x][y] < 220:
            crop_sum += crop[x][y]
            crop_count += 1
      if crop_count != 0:
        avg = crop_sum / crop_count
        depth = avg / ratio
      else:
        depth = 255 / ratio
    """
    crop = cv2.applyColorMap(crop, cv2.COLORMAP_JET)
    cv2.imshow('crop', crop)
    print(depth)
    cv2.waitKey(0)
    """
    return [depth]

# ==================================================================================================================

# loading average dimension
dimension_map = {}
f = open('utils/class_averages.txt', 'r')
dimension_map = json.load(f)
for class_ in dimension_map:
  dimension_map[class_]['total'] = np.asarray(dimension_map[class_]['total'])

def get_dim(class_):
  class_ = class_.lower()
  if class_ == 'dontcare':
    return [0.0, 0.0, 0.0]
  else:
    return dimension_map[class_]['total'] / dimension_map[class_]['count']

# ==================================================================================================================

def loading_dets(img_dir, name, disparity, depth_ratio):

  # loading label -> alpha, bbox
  dets = {}
  dets[1], dets[2] = [], []
  f = open(img_dir + 'label_2/' + name + '.txt', 'r')
  for line in f.readlines():
    line_str = line.strip().split(' ')
    line_float = np.array(line_str[3:], dtype=np.float32)
    w = line_float[3] - line_float[1]
    h = line_float[4] - line_float[2]
    center = np.array([line_float[1] + w / 2, line_float[2] + h / 2])
    score  = np.array([1.0])
    alpha  = np.array([line_float[0]])
    depth  = np.array(get_depth(disparity, center, w, h, depth_ratio))
    dimens = np.array(get_dim(line_str[0])) #np.array(line_float[5:8])
    wh     = np.array([w, h])
    det = np.array(np.concatenate((center, score, alpha, depth, dimens, wh)))
    if line_str[0] == 'Pedestrian':
      dets[1].append(det)
    elif line_str[0] == 'Car':
      dets[2].append(det)
  dets[1], dets[2] = np.array(dets[1]), np.array(dets[2])

  return dets

# ==================================================================================================================

def loading_calib(img_dir, name):
  # loading calib
  f = open(img_dir + 'calib/' + name + '.txt', 'r')
  for line in f.readlines():
    line_str = line.strip().split(' ')
    if line_str[0] == 'P2:':
      calib = np.array(line_str[1:], dtype=np.float32).reshape(3, 4)

  return calib

