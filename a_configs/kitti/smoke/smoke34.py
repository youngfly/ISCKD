_base_ = [
    './kitti-mono3d.py'
]
model = dict(
    type='SMOKEMono3D',
    backbone=dict(
        type='DLANet',
        depth=34,
        in_channels=3,
        norm_cfg=dict(type='GN', num_groups=32),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/pretrained model/dla34-ba72cf86.pth'
        )
    ),
    neck=dict(
        type='DLANeck',
        in_channels=[16, 32, 64, 128, 256, 512],
        start_level=2,
        end_level=5,
        norm_cfg=dict(type='GN', num_groups=32)),
    bbox_head=dict(
        type='SMOKEMono3DHead',
        num_classes=3,
        in_channels=64,
        dim_channel=[3, 4, 5],
        ori_channel=[6, 7],
        stacked_convs=0,
        feat_channels=64,
        use_direction_classifier=False,
        diff_rad_by_sin=False,
        pred_attrs=False,
        pred_velo=False,
        dir_offset=0,
        strides=None,
        group_reg_dims=(8, ),
        cls_branch=(256, ),
        reg_branch=((256, ), ),
        num_attrs=0,
        bbox_code_size=7,
        dir_branch=(),
        attr_branch=(),
        bbox_coder=dict(
            type='SMOKECoder',
            base_depth=(28.01, 16.32),
            base_dims=((0.88, 1.73, 0.67), (1.78, 1.70, 0.58), (3.88, 1.63,
                                                                1.53)),
            code_size=7),
        loss_cls=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', reduction='sum', loss_weight=1 / 300),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_attr=None,
        conv_bias=True,
        dcn_on_last_conv=False),
    train_cfg=None,
    test_cfg=dict(topK=100, local_maximum_kernel=3, max_per_img=100))

# optimizer
optimizer = dict(type='Adam', lr=2.5e-4)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', warmup=None, step=[50])

find_unused_parameters = True
class_names = ['Pedestrian', 'Cyclist', 'Car']
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
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='RandomShiftScale', shift_scale=(0.2, 0.4), aug_prob=0.3),
    dict(type='AffineResize', img_scale=(1280, 384), down_ratio=4),
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
        img_scale=(1280, 384),
        flip=False,
        transforms=[
            dict(type='AffineResize', img_scale=(1280, 384), down_ratio=4),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=72)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
checkpoint_config = dict(interval=1)
# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook

# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
