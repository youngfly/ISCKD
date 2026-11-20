# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import DETECTORS
from .fcos_mono3d import FCOSMono3D
from .smoke_mono3d import SMOKEMono3D
from mmdet.models import build_detector
from mmdet.models import build_loss
from mmdet.utils import get_root_logger
from mmdet.core import bbox_overlaps
from mmcv.runner import load_checkpoint
import torch
import math
from mmdet3d.core import build_anchor_generator
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import matplotlib
import mmcv
import numpy as np


def levels_to_images(mlvl_tensor):
    """Concat multi-level feature maps by image.

    [feature_level0, feature_level1...] -> [feature_image0, feature_image1...]
    Convert the shape of each element in mlvl_tensor from (N, C, H, W) to
    (N, H*W , C), then split the element to N elements with shape (H*W, C), and
    concat elements in same image of all level along first dimension.

    Args:
        mlvl_tensor (list[torch.Tensor]): list of Tensor which collect from
            corresponding level. Each element is of shape (N, C, H, W)

    Returns:
        list[torch.Tensor]: A list that contains N tensors and each tensor is
            of shape (num_elements, C)
    """
    batch_size = mlvl_tensor[0].size(0)
    batch_list = [[] for _ in range(batch_size)]
    channels = mlvl_tensor[0].size(1)
    for t in mlvl_tensor:
        t = t.permute(0, 2, 3, 1)
        t = t.view(batch_size, -1, channels).contiguous()
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [torch.cat(item, 0) for item in batch_list]


@DETECTORS.register_module()
class DIS_SMOKE(SMOKEMono3D):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 tea_model,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 tea_pretrained=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     ratios=[1.0],
                     octave_base_scale=8,
                     scales_per_octave=1,
                     strides=[8, 16, 32, 64]),
                 ):
        super(DIS_SMOKE, self).__init__(backbone, neck, bbox_head, train_cfg,
                                         test_cfg, pretrained)
        # todo 构建教师模型
        self.tea_model = build_detector(tea_model, train_cfg=train_cfg, test_cfg=test_cfg)
        # todo 加载教师模型的权重---全部权重
        self.load_weights(pretrained=tea_pretrained)
        # todo 教师模型仅仅加载backbone
        self.freeze_models()

        # todo 采用smooth l1 loss 作为深度蒸馏损失函数
        loss_dis_depth_cfg = dict(type='SmoothL1Loss', reduction='none', loss_weight=1.0)
        loss_dis_depth_cfg2 = dict( type='UncertainSmoothL1Loss', alpha=1.0, beta=3.0, loss_weight=1.0),
        self.loss_dis_depth = build_loss(loss_dis_depth_cfg)

        # todo 采用smoothL1 loss 作为回归蒸馏的损失函数
        loss_dis_bbox_cfg = dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)
        self.loss_dis_bbox = build_loss(loss_dis_bbox_cfg)

        # todo 采用Focal loss 作为分类里面的蒸馏损失函数
        # loss_dis_cls_cfg=dict( type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0)
        # self.loss_dis_bbox = build_loss(loss_dis_cls_cfg)

        # todo 针对学生的fpn的特征层的尺度变为原来的一半，此处从configs获取学生fpn通道数
        self.in_channels_adapt = bbox_head.in_channels
        self.out_channels = 64
        self.adapt_conv = nn.Conv2d(self.in_channels_adapt, self.out_channels, 1, padding=0)

        # todo 生成anchor
        self.anchor_generator = build_anchor_generator(anchor_generator)
        self.num_anchors = self.anchor_generator.num_base_anchors[0]

        # todo 蒸馏权重
        self.cls_weight = 100
        self.fea_weight = 100

    # todo 冻结教师模型
    def freeze_models(self):
        self.tea_model.eval()
        for param in self.tea_model.parameters():
            param.requires_grad = False

    # todo 载入模型
    def load_weights(self, pretrained=None):
        # todo 加载教师模型
        if isinstance(pretrained, str):
            print("begin load the teacher model")
            logger = get_root_logger()
            load_checkpoint(self.tea_model, pretrained, strict=False, logger=logger)
            print("load teacher model success")
        else:
            raise TypeError('pretrained must be a str')

    # todo 前向训练
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      centers2d,
                      depths,
                      attr_labels=None,
                      gt_bboxes_ignore=None):

        # todo 获取教师网络的输出
        self.tea_model.eval()
        with torch.no_grad():
            tea_x = self.tea_model.extract_feat(img)
            tea_outs = self.tea_model.bbox_head(tea_x)
            tea_loss_inputs = tea_outs + (gt_bboxes, gt_labels, gt_bboxes_3d, gt_labels_3d,
                                          centers2d, depths, attr_labels, img_metas)
            # tea_losses = self.bbox_head.loss(*tea_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        # todo 获取学生的网络的输出
        stu_x = self.extract_feat(img)
        stu_outs = self.bbox_head(stu_x)
        loss_inputs = stu_outs + (gt_bboxes, gt_labels, gt_bboxes_3d, gt_labels_3d,
                                          centers2d, depths, attr_labels, img_metas)
        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        # todo tupe [7, [4, [n, c, h ,w]]], 3代表三个输出的分支头， 5代表5个尺度，n代表batch_size,
        # todo 7: 7个预测组, 4:4个level, n:batchsize, c:channel, h,w:
        # todo 7: cls_score[3], bbox_pred[27], dir_cls_pred[2], depth_cls_pred[8], weight[1], attr_pred, centerness[1]
        # todo [h,w] 312, 156, 78 ,39

        # todo 学生的特征进行提前通道变换
        stu_adapt = self.adapt_old(stu_x)

        # todo 学生网络输出结果
        # pos_inds, pos_flatten_cls, \
        # pos_bbox_preds, pos_dir_cls_preds, \
        # pos_depth_cls_preds, pos_depth_stu, \
        # pos_prob_depth_preds_stu, \
        # pos_fuse_depth, gt_depth, faltten_cls_stu = self.bbox_head.loss_stu(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        # todo 教师网络输出
        # pos_inds_tea, pos_flatten_cls_tea, \
        # pos_bbox_preds_tea, pos_dir_cls_preds_tea, \
        # pos_depth_cls_preds_tea, pos_depth_tea, \
        # pos_prob_depth_preds_tea,\
        # pos_fuse_depth_tea, _,  faltten_cls_tea= self.tea_model.bbox_head.loss_stu(*tea_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        # todo 透视原理蒸馏
        # todo 计算每个点到中心的距离，首先计算中心点的坐标，根据奇数偶数来做区分, stu_x = [4, [3, [128, h, w]]]
        # todo dt_mat =[4, [h , w]]
        A = 0.3
        sigma_x = 0.2
        sigma_y = 0.2
        dt_mat = []
        for i in range(len(stu_x)):
            dt_level = torch.zeros(stu_x[i][0].shape[1], stu_x[i][0].shape[2])
            h = dt_level.shape[0]
            w = dt_level.shape[1]
            if w % 2 == 0:
                w_cen = w / 2
            elif w % 2 == 1:
                w_cen = (w - 1) / 2
            if h % 2 == 0:
                h_cen = h / 2
            elif h % 2 == 1:
                h_cen = (h - 1) / 2
            # todo 距离需要归一化
            for j in range(h):
                for k in range(w):
                    dt_level[j][k] = A * math.exp(- (k/w - w_cen/w) ** 2 / sigma_x - (j/h - h_cen/h) ** 2 / sigma_y)
            # todo 添加一个新的维度，可以理解为通道维度
            dt_level = dt_level.unsqueeze(0).unsqueeze(0)
            dt_mat.append(dt_level)

        # todo 展平特征关系
        # todo 特征层的样本选取*********************************
        # todo 获取教师和学生特征层的信息，展平方便提取关键信息
        ft = levels_to_images(tea_x)
        ft_flatten = torch.cat(ft, 0).view(-1, ft[0].size(-1))
        # ft_flatten_pos = ft_flatten[pos_inds]
        fs = levels_to_images(stu_adapt)
        fs_flatten = torch.cat(fs, 0).view(-1, fs[0].size(-1))
        # fs_flatten_pos = fs_flatten[pos_inds]

        # todo 采用透视矩阵
        dt = levels_to_images(dt_mat)
        dt_flatten = torch.cat(dt, 0).view(-1, dt[0].size(-1))
        dt_flatten_all = dt_flatten.repeat(len(stu_x[0]), 1)
        dt_flatten_all = dt_flatten_all.squeeze(1).cuda(0)

        # todo 采用FE矩阵
        ft_attention, _ = torch.max(ft_flatten, dim=1)
        fs_attention, _ = torch.max(fs_flatten, dim=1)
        # todo get the weight
        FE_attention = torch.abs(torch.pow(ft_attention - fs_attention, 2))
        FE = FE_attention

        # todo FitNet 特征层 mask_normal
        # mask_normal = torch.ones_like(FE)

        # todo FGFI 特征层 fgfi_mask 最终需要展平
        # fgfi_mask_old = self.get_fgfi_masks(stu_x, gt_bboxes, img_metas)
        # fgfi_mask = self.mask_transform(fgfi_mask_old)
        # fgfi_mask = fgfi_mask.squeeze(1)

        # todo TAR 高斯特诊层 tar_mask + 分类 最终需要展平
        # tar_mask_old = self.get_tar_masks(stu_x, gt_bboxes, img_metas, img)
        # tar_mask = self.mask_transform(tar_mask_old)
        # tar_mask = tar_mask.squeeze(1)

        # todo GID 实例蒸馏 GID_maks + 分类 最终需要展平 可以参考DIL中的实例获取
        # gid_mask = torch.ones_like(pos_inds)

        # todo 知识蒸馏**************************************************
        # todo Fea ******************************
        # todo FitNet
        # sup_stu_fea = self.dis_fea_ada(ft_flatten, fs_flatten, mask_normal)
        # todo FGFI
        # sup_stu_fea = self.dis_fea_ada(ft_flatten, fs_flatten, fgfi_mask)
        # todo TAR
        # sup_stu_fea = self.dis_fea_ada(ft_flatten, fs_flatten, tar_mask)
        # todo GID
        # sup_stu_fea = self.dis_fea_ada(ft_flatten_pos, fs_flatten_pos, gid_mask)
        # todo FE_attention
        # sup_stu_fea = self.dis_fea_ada(ft_flatten, fs_flatten, FE)

        # todo ours
        # todo gauss
        sup_stu_fea = self.dis_fea_ada(ft_flatten, fs_flatten, dt_flatten_all) * self.fea_weight
        losses.update(dict(loss_stu_fea=sup_stu_fea))

        # todo cls ******************************
        # tem = 1
        # todo KD old
        # sup_stu_cls = self.dis_cls(mask_normal, ft_flatten, fs_flatten, tem)
        # todo KD new
        # sup_stu_cls = self.dis_KL(mask_normal, faltten_cls_stu, faltten_cls_tea, tem)
        # sup_stu_cls = self.dis_cls(mask_normal, faltten_cls_stu, faltten_cls_tea, tem)

        # todo GID
        # sup_stu_cls = self.dis_cls(gid_mask, pos_flatten_cls, pos_flatten_cls_tea, tem)
        # todo TAR
        # sup_stu_cls = self.dis_cls(gid_mask, pos_flatten_cls, pos_flatten_cls_tea, tem)

        # todo Ours
        # losses.update(dict(loss_stu_cls=sup_stu_cls))

        # todo reg *******************************
        # todo offset  pos_bbox_preds[:, :2]   pos_bbox_preds_tea[:, :2] SmoothL1Loss
        # sup_stu_offset = self.loss_dis_bbox(pos_bbox_preds[:, :2], pos_bbox_preds_tea[:, :2])
        # todo size   pos_bbox_preds[:, 3:6]   pos_bbox_preds_tea[:, 3:6] SmoothL1Loss
        # sup_stu_size = self.loss_dis_bbox(pos_bbox_preds[:, 3:6], pos_bbox_preds_tea[:, 3:6])
        # todo rot    pos_bbox_preds[:, 6]   pos_bbox_preds_tea[:, 6] SmoothL1Loss
        # sup_stu_rot = self.loss_dis_bbox(pos_bbox_preds[:, 6], pos_bbox_preds_tea[:, 6])

        # losses.update(dict(loss_stu_offset=sup_stu_offset))
        # losses.update(dict(loss_stu_size=sup_stu_size))
        # losses.update(dict(loss_stu_rot=sup_stu_rot))

        # todo depth ******************************
        # todo distance weight
        # pos_mask = torch.ones_like(pos_inds)
        # todo use stu
        # D_mask = pos_mask * (pos_depth_stu / torch.max(pos_depth_stu))
        # todo use tea
        # D_mask = pos_mask * (pos_depth_tea / torch.max(pos_depth_tea))
        # todo use average
        # D_mask = pos_mask * (pos_depth_stu + pos_depth_tea / torch.max(pos_depth_stu) + torch.max(pos_depth_tea))
        # todo use gt
        # D_mask = pos_mask * (gt_depth / torch.max(gt_depth))

        # todo weight function
        # todo exp - 1
        # mask_func = torch.exp(D_mask) - 1
        # todo sin + 1
        # mask_func = torch.sin(D_mask) + 1
        # todo 2sin
        # mask_func = 2 * torch.sin(D_mask)

        # todo 将深度距离归一化来加快收敛
        # p_d_s_norm = pos_depth_stu / torch.max(gt_depth)
        # p_d_t_norm = pos_depth_tea / torch.max(gt_depth)

        # todo distill direct depth 可选的mask, pos_mask:全部的都是一, mask_func:深度加权
        # sup_stu_depth_DR = self.dis_depth(mask_func, pos_depth_stu, pos_depth_tea)
        # todo disitll pro depth
        # sup_stu_depth_DP = self.dis_depth(mask_func, pos_prob_depth_preds_stu, pos_prob_depth_preds_tea)
        # todo distill fuse depth
        # sup_stu_depth_fuse = self.dis_depth(mask_func, pos_fuse_depth, pos_fuse_depth_tea)
        # sup_stu_depth_DR = self.dis_depth(mask_func, p_d_s_norm, p_d_t_norm)

        # losses.update(dict(loss_stu_depth_DR=sup_stu_depth_DR))
        # losses.update(dict(loss_stu_depth_DP=sup_stu_depth_DP))
        # losses.update(dict(loss_stu_depth_fuse=sup_stu_depth_fuse))
        return losses

    # todo some masks
    def get_fgfi_masks(self, fea, gt_bboxes, img_metas):
        featmap_sizes = [featmap.size()[-2:] for featmap in fea]
        device = fea[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        mask_batch = []
        A = self.num_anchors
        batch_size = len(gt_bboxes)
        levels = len(anchor_list[0])
        for i in range(batch_size):
            mask_per_im = []
            for j in range(levels):
                height, width = featmap_sizes[j][0], featmap_sizes[j][1]
                all_anchors = anchor_list[i][j]
                # todo 由于部分的切片里面存在着无标签的情况，跳过这个部分
                if gt_bboxes[i].shape[0] == 0:
                    continue
                # todo
                IOU_map = bbox_overlaps(all_anchors, gt_bboxes[i]).view(height, width, A, gt_bboxes[i].shape[0])
                max_iou, _ = torch.max(IOU_map.view(height * width * A, gt_bboxes[i].shape[0]), dim=0)
                mask_per_level = torch.zeros([height, width], dtype=torch.int64).cuda()
                for k in range(gt_bboxes[i].shape[0]):
                    if torch.sum(gt_bboxes[i][k]) == 0.:
                        break
                    max_iou_per_gt = max_iou[k] * 0.5
                    mask_per_gt = torch.sum(IOU_map[:, :, :, k] > max_iou_per_gt, dim=2)
                    mask_per_level += mask_per_gt
                mask_per_im.append(mask_per_level)
            mask_batch.append(mask_per_im)

        return mask_batch

    def get_tar_masks(self, cls_scores, gt_bboxes, img_metas, img):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
        # todo 获取一个批次图片的尺寸
        img_ori = img.size()[-2:]
        img_ori = torch.tensor([img_ori])
        # todo img_ori=[1024,1024]
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        mask_batch = []

        batch_size = len(gt_bboxes)
        levels = len(anchor_list[0])
        for i in range(batch_size):
            mask_per_im = []
            # todo gt_bboxes=[gt_num,4]
            # todo 根据gt_box计算其面积
            area = torch.sqrt((gt_bboxes[i][:, 2] - gt_bboxes[i][:, 0]) * (gt_bboxes[i][:, 3] - gt_bboxes[i][:, 1]))
            # todo arae=[gt_num], 值=[0~1024]
            # todo 计算原始图片的尺寸
            area_image = torch.sqrt((img_ori[:, 0] * img_ori[:, 1]).to(torch.float32)).cuda()
            # todo area_image=[1],值=[1024]
            # todo 获得不同尺寸的gt应该分配给不同的层级,类似于fpn的操作
            target_lvls = torch.floor(4 + torch.log2(area / area_image + 1e-6)).cuda()
            # todo 将level限制5个fpn层级
            target_lvls = target_lvls.clamp(min=0, max=4).long()
            # todo target_lvls=[gt_num], 值=[0~4]
            for j in range(levels):
                height, width = featmap_sizes[j][0], featmap_sizes[j][1]
                mask_per_level = torch.zeros([height, width], dtype=torch.int64).cuda()
                # todo 获取每个特征图的尺寸
                fea_scale = featmap_sizes[j]
                # todo 将特征图尺寸转换为tensor
                fea_scale = torch.tensor([fea_scale])
                # todo fea_scale=[1,1] 值=[128,128],[64,64],[32,32],[16,16],[8,8]
                # todo 计算出每一层特征图相对于原始图片的缩放比率。
                # todo 对于dota刚好是长款等比，但是nwpu长宽比例不同，可能后面需要调整顺序
                percent = torch.div(fea_scale.float(), img_ori.float()).cuda()
                # todo percent=[1,1] 值=[0.125], [0.0625],[0.0312],[0.0156],[0.0078]
                # print(percent)
                # todo 构建按比例缩放后的gt坐标组
                gt_trans = torch.zeros(gt_bboxes[i].shape, dtype=torch.float32).cuda()
                gt_trans[:, :2] = gt_bboxes[i][:, :2] * percent
                gt_trans[:, 2:] = gt_bboxes[i][:, 2:] * percent
                # todo gt坐标放缩到对应的特征图上的位置
                # todo gt_mask尺寸为和gt_bboxes尺寸相同，gt_mask=[gt_num,4]
                gt_mask = torch.zeros(gt_bboxes[i].shape, dtype=torch.float32).cuda()
                # todo 根据当前图片里每所有的gt，生成对应当前层级的mask
                for k in range(gt_bboxes[i].shape[0]):
                    # todo 如果第k个gt分配的level层级等于当前的level层级，则将该层级的gt的gt_mask置为1
                    if target_lvls[k] == j:
                        gt_mask[k, :] = 1
                # todo 将mask和gt_trans相乘，得到对应的gt
                gt_trans = gt_trans * gt_mask
                # todo 根据不同的层级
                for k in range(gt_bboxes[i].shape[0]):
                    # todo 如果
                    if torch.sum(gt_bboxes[i][k]) == 0.:
                        break
                    # todo 为该层级生成特征图尺寸的mask
                    mask_per_gt = torch.zeros(height, width, dtype=torch.int64).cuda()
                    # todo 将mask 里面gt位置的值置为1
                    mask_per_gt[gt_trans[k][1].int():gt_trans[k][3].int(),
                    gt_trans[k][0].int():gt_trans[k][2].int()] = 1
                    # todo 将这个level里面所有的gt位置叠加起来，得到一个层级特征mask
                    mask_per_level += mask_per_gt
                # todo 得到一张图片的特征mask
                mask_per_im.append(mask_per_level)
            # todo 得到一个batch所有的mask
            mask_batch.append(mask_per_im)
        return mask_batch

    # todo distill loss
    def dis_fea_ada(self, ft, fs, mask):
        loss_fea = torch.sqrt((torch.pow(ft - fs, 2) * mask.unsqueeze(1)).sum())
        sup_loss = loss_fea / len(ft)
        return sup_loss

    def dis_cls(self, mask_A, pos_cls_stu, pos_cls_tea, T):
        loss_cls = torch.sqrt((torch.pow(pos_cls_stu/T - pos_cls_tea/T, 2) * mask_A.unsqueeze(1)).sum())
        sup_loss = loss_cls / len(pos_cls_stu)
        return sup_loss

    def dis_KL(self, mask_A, stu_cls, tea_cls, KL):
        kl_loss = F.kl_div(stu_cls.softmax(dim=-1).log(), tea_cls.softmax(dim=-1), reduction='mean')
        return kl_loss

    def dis_depth(self, mask, depth_stu, depth_tea):
        loss_dep = torch.sum(self.loss_dis_depth(depth_stu, depth_tea) * mask)
        sup_loss = loss_dep / len(depth_stu)

        return sup_loss

    # todo some tools
    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        num_imgs = len(img_metas)
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.anchor_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def mask_transform(self, fgfi_mask_old):
        mask_all = torch.tensor([]).cuda()
        for i in range(len(fgfi_mask_old)):
            for l in range(len(fgfi_mask_old[i])):
                mask_mini = fgfi_mask_old[i][l].view(-1, 1)
                mask_all = torch.cat((mask_all, mask_mini), 0)

        return mask_all

    def adapt_old(self, x):
        stu_adapt = []
        for i in range(len(x)):
            stu_ada = []
            stu_ada = self.adapt_conv(x[i])
            # print(stu_ada.shape)
            stu_adapt.append(stu_ada)
        return stu_adapt