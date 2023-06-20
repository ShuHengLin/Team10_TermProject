import os
import cv2
import numpy as np
from tqdm import tqdm
from utils.data_loading import loading_calib, loading_dets
from utils.project import ddd_post_process_3d
from utils.draw import drawing

# ==================================================================================================================

out_dir = 'outputs/outputs_DenseDepth/'
img_dir = 'kitti_data/training/'
ls = os.listdir(img_dir + 'image_2')

for img_name in tqdm(sorted(ls)):

  name = img_name.strip().split('.png')[0]

  img  = cv2.imread(img_dir + 'image_2/' + img_name)

  disparity = np.load('DenseDepth/disparity/' + name + '.npy')
  depth_ratio = 3.26160710191854

  dets  = loading_dets(img_dir, name, disparity, depth_ratio)
  calib = loading_calib(img_dir, name)

  dets = ddd_post_process_3d([dets], calib)
  results = dets[0]

  for cat in results:
    for i in range(len(results[cat])):
      dim   = results[cat][i, 5:8]
      loc   = results[cat][i, 8:11]
      rot_y = results[cat][i, 11]
      if loc[2] > 1:
        img = drawing(img, dim, loc, rot_y, calib, cat)

# ==================================================================================================================

  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
  cv2.imwrite(out_dir + name + '.jpg',           img,       [cv2.IMWRITE_JPEG_QUALITY, 90])
  cv2.imwrite(out_dir + name + '_disparity.jpg', disparity, [cv2.IMWRITE_JPEG_QUALITY, 90])

#  cv2.imshow(name, img)
#  cv2.imshow('disparity', disparity)
#  cv2.waitKey(0)
#  cv2.destroyAllWindows()

  f = open(out_dir + name + '.txt', 'w')
  for cat in results:
    for i in range(len(results[cat])):
      out_type = 'car' if cat == 2 else 'pedestrian'
      out_trunc = ' 0.00'
      out_occlu = ' 0'
      out_alpha = ' ' + str(results[cat][i, 0])
      out_bbox  = ' ' + str(results[cat][i, 1]) + ' ' + str(results[cat][i, 2]) + ' ' + str(results[cat][i, 3]) + ' ' + str(results[cat][i, 4])
      out_dimen = ' ' + str(results[cat][i, 5]) + ' ' + str(results[cat][i, 6]) + ' ' + str(results[cat][i, 7])
      out_locat = ' ' + str(results[cat][i, 8]) + ' ' + str(results[cat][i, 9]) + ' ' + str(results[cat][i, 10])
      out_rotat = ' ' + str(results[cat][i, 11])
      out_score = ' ' + str(1.0)
      f.write(out_type + out_trunc + out_occlu + out_alpha + out_bbox + out_dimen + out_locat + out_rotat + out_score + '\n')

