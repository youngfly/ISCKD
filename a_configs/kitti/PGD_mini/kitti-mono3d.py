dataset_type = 'KittiMonoDataset'
data_root = '/home/data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
input_modality = dict(use_lidar=False, use_camera=True)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='Resize', img_scale=(1242, 375), keep_ratio=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_3d', 'gt_labels_3d',
            'centers2d', 'depths'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1242, 375),
        flip=False,
        transforms=[
            dict(type='RandomFlip3D'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img']),
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['img'])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_train_mono3d.coco.json',
        info_file=data_root + 'kitti_infos_train.pkl',
        img_prefix=data_root,
        classes=class_names,
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        box_type_3d='Camera'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val_mono3d.coco.json',
        info_file=data_root + 'kitti_infos_val.pkl',
        img_prefix=data_root,
        classes=class_names,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        box_type_3d='Camera'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_test_mono3d.coco.json',
        info_file=data_root + 'kitti_infos_test.pkl',
        img_prefix=data_root,
        classes=class_names,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        box_type_3d='Camera'))
evaluation = dict(interval=2)

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

checkpoint_config = dict(interval=1)
# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
# work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

# todo sota config
# work_dir = "work_dirs/Dis_PGD_r18h_KD_mask"
# work_dir = "work_dirs/Dis_PGD_r18h_Fitnet_mask"
# work_dir = "work_dirs/Dis_PGD_r18h_FGFI_mask"
# work_dir = "work_dirs/Dis_PGD_r18h_TAR_mask"
# work_dir = "work_dirs/Dis_PGD_r18h_GID_mask"
# work_dir = "work_dirs/Dis_PGD_r18h_DeFeat_mask"
# work_dir = "work_dirs/Dis_PGD_r18h_AT_mask"
# work_dir = "work_dirs/Dis_PGD_r18h_fgd_mask"

# work_dir = "work_dirs/Dis_PGD_r18_KD_mask"
# work_dir = "work_dirs/Dis_PGD_r18_Fitnet_mask"
# work_dir = "work_dirs/Dis_PGD_r18_FGFI_mask"
# work_dir = "work_dirs/Dis_PGD_r18_TAR_mask"
# work_dir = "work_dirs/Dis_PGD_r18_GID_mask"
# work_dir = "work_dirs/Dis_PGD_r18_DeFeat_mask"
# work_dir = "work_dirs/Dis_PGD_r18_AT_mask"
# work_dir = "work_dirs/Dis_PGD_r18_fgd_mask"

# work_dir = "work_dirs/Dis_PGD_reg800h_KD_mask"
# work_dir = "work_dirs/Dis_PGD_reg800h_FitNet_mask"
# work_dir = "work_dirs/Dis_PGD_reg800h_FGFI_mask"
# work_dir = "work_dirs/Dis_PGD_reg800h_TAR_mask"
# work_dir = "work_dirs/Dis_PGD_reg800h_GID_mask3"
# work_dir = "work_dirs/Dis_PGD_reg800h_DeFeat_mask3"
# work_dir = "work_dirs/Dis_PGD_reg800h_AT_mask"
# work_dir = "work_dirs/Dis_PGD_reg800h_fgd_mask4"

# work_dir = "work_dirs/Dis_PGD_reg800_KD_mask"
# work_dir = "work_dirs/Dis_PGD_reg800_FitNet_mask"
# work_dir = "work_dirs/Dis_PGD_reg800_FGFI_mask"
# work_dir = "work_dirs/Dis_PGD_reg800_TAR_mask"
# work_dir = "work_dirs/Dis_PGD_reg800_GID_mask5"
# work_dir = "work_dirs/Dis_PGD_reg800_DeFeat_mask3"
# work_dir = "work_dirs/Dis_PGD_reg800_AT_mask5"
# work_dir = "work_dirs/Dis_PGD_reg800_fgd_mask4"

# work_dir = "work_dirs/Dis_PGD_mobv2_KD_mask"
# work_dir = "work_dirs/Dis_PGD_mobv2_FitNet_mask"
# work_dir = "work_dirs/Dis_PGD_mobv2_FGFI_mask"
# work_dir = "work_dirs/Dis_PGD_mobv2_TAR_mask"
# work_dir = "work_dirs/Dis_PGD_mobv2_GID_mask"
# work_dir = "work_dirs/Dis_PGD_mobv2_DeFeat_mask"
# work_dir = "work_dirs/Dis_PGD_mobv2_AT_mask"

# work_dir = "work_dirs/Dis_PGD_shuf_KD_mask"
# work_dir = "work_dirs/Dis_PGD_shuf_FitNet_mask"
# work_dir = "work_dirs/Dis_PGD_shuf_FGFI_mask"
# work_dir = "work_dirs/Dis_PGD_shuf_TAR_mask"
# work_dir = "work_dirs/Dis_PGD_shuf_GID_mask"
# work_dir = "work_dirs/Dis_PGD_shuf_DeFeat_mask"
# work_dir = "work_dirs/Dis_PGD_shuf_AT_mask"

# todo baseline
# work_dir = "work_dirs/PGD_r18"
# work_dir = "work_dirs/PGD_shuf2"
# work_dir = "work_dirs/PGD_mobv2"
# work_dir = "work_dirs/PGD_reg800"

# todo cls & ori distilllation
# work_dir = "work_dirs/Dis_PGD_r18h_cls500_L2_all_dep"
# work_dir = "work_dirs/Dis_PGD_r18h_cls500_L2_all_exp"
# work_dir = "work_dirs/Dis_PGD_r18h_cls500_L2_all_exp*dep"
# work_dir = "work_dirs/Dis_PGD_r18h_cls500_L2_all_lin"
# work_dir = "work_dirs/Dis_PGD_r18h_cls500_L2_all_lin*dep"

# work_dir = "work_dirs/Dis_PGD_r18h_ori1000_L2_pos"
# work_dir = "work_dirs/Dis_PGD_r18h_ori1000_KL_pos"
# work_dir = "work_dirs/Dis_PGD_r18h_ori1000_L2_pos_dep"
# work_dir = "work_dirs/Dis_PGD_r18h_ori1000_L2_pos_exp"
# work_dir = "work_dirs/Dis_PGD_r18h_ori1000_L2_pos_exp*dep"
# work_dir = "work_dirs/Dis_PGD_r18h_ori1000_L2_pos_lin"
# work_dir = "work_dirs/Dis_PGD_r18h_ori1000_L2_pos_lin*dep"

# todo demo
# work_dir = "work_dirs/demo"

# todo combine
# work_dir = "work_dirs/Dis_PGD_r18h_cls1000_L2_all_exp*dep_fea3"
# work_dir = "work_dirs/Dis_PGD_r18_fea_cls1000_KL_all_fea2_0.7"
# work_dir = "work_dirs/Dis_PGD_r18_fea5_0.2"
work_dir = "work_dirs/Dis_PGD_reg800_fea_l2_cls"