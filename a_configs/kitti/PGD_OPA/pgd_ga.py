_base_ = ['./pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d.py']
model = dict(bbox_head=dict(type='PGD_GA_HEAD'))
checkpoint_config = dict(interval=4)
# work_dir = 'work_dirs/pgd_t15_150_100_ri_0.5_0.7'
work_dir = 'work_dirs/demo'