import os
import numpy as np
import json

jsonpath = '/home/data/kitti/kitti_infos_train_mono3d.coco.json'
# jsonpath = '/home/data/kitti/kitti_infos_val_mono3d.coco.json'


depth_dic = {}
ad = np.zeros((29, 2))
with open(jsonpath, 'r') as f:
    # todo annotations images categories
    label = json.load(f)
    annos = label["annotations"]
    for anno in annos:
        depth = anno['center2d'][2]
        bin = int(depth / 3)
        if bin not in depth_dic:
            depth_dic.update({"{}".format(bin): 0})
    for anno in annos:
        file_name = anno['file_name']
        area = anno['area']
        category = anno['category_id']
        depth = anno['center2d'][2]
        bin = int(depth / 3)
        depth_dic["{}".format(bin)] += 1
        # todo 计算面积和深度的关系
        ad[bin][0] = ad[bin][0] + area
        ad[bin][1] = ad[bin][1] + 1
print(depth_dic)
dir_sort = sorted(depth_dic.items(), key=lambda x:x[0])
print(dir_sort)
print(ad)
# todo
array3 = ad[:, 0] / ad[:, 1]
print(array3)