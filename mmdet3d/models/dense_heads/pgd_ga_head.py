import numpy as np
import torch
from mmcv.cnn import Scale, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core import box3d_multiclass_nms, xywhr2xyxyr
from mmdet3d.core.bbox import points_cam2img, points_img2cam
from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import axis_aligned_bbox_overlaps_3d
from mmdet.core import distance2bbox, multi_apply
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.models.builder import HEADS, build_loss
from .fcos_mono3d_head import FCOSMono3DHead
import matplotlib.pyplot as plt
import geatpy as ea
from scipy.optimize import minimize
INF = 1e8

try:
    import sklearn.mixture as skm
except ImportError:
    skm = None

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


@HEADS.register_module()
class PGD_GA_HEAD(FCOSMono3DHead):
    r"""Anchor-free head used in `PGD <https://arxiv.org/abs/2107.14160>`_.

    Args:
        use_depth_classifer (bool, optional): Whether to use depth classifier.
            Defaults to True.
        use_only_reg_proj (bool, optional): Whether to use only direct
            regressed depth in the re-projection (to make the network easier
            to learn). Defaults to False.
        weight_dim (int, optional): Dimension of the location-aware weight
            map. Defaults to -1.
        weight_branch (tuple[tuple[int]], optional): Feature map channels of
            the convolutional branch for weight map. Defaults to ((256, ), ).
        depth_branch (tuple[int], optional): Feature map channels of the
            branch for probabilistic depth estimation. Defaults to (64, ),
        depth_range (tuple[float], optional): Range of depth estimation.
            Defaults to (0, 70),
        depth_unit (int, optional): Unit of depth range division. Defaults to
            10.
        division (str, optional): Depth division method. Options include
            'uniform', 'linear', 'log', 'loguniform'. Defaults to 'uniform'.
        depth_bins (int, optional): Discrete bins of depth division. Defaults
            to 8.
        loss_depth (dict, optional): Depth loss. Defaults to dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0).
        loss_bbox2d (dict, optional): Loss for 2D box estimation. Defaults to
            dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0).
        loss_consistency (dict, optional): Consistency loss. Defaults to
            dict(type='GIoULoss', loss_weight=1.0),
        pred_velo (bool, optional): Whether to predict velocity. Defaults to
            False.
        pred_bbox2d (bool, optional): Whether to predict 2D bounding boxes.
            Defaults to True.
        pred_keypoints (bool, optional): Whether to predict keypoints.
            Defaults to False,
        bbox_coder (dict, optional): Bounding box coder. Defaults to
            dict(type='PGDBBoxCoder', base_depths=((28.01, 16.32), ),
            base_dims=((0.8, 1.73, 0.6), (1.76, 1.73, 0.6), (3.9, 1.56, 1.6)),
            code_size=7).
    """

    def __init__(self,
                 use_depth_classifier=True,
                 use_onlyreg_proj=False,
                 weight_dim=-1,
                 weight_branch=((256, ), ),
                 depth_branch=(64, ),
                 depth_range=(0, 70),
                 depth_unit=10,
                 division='uniform',
                 depth_bins=8,
                 loss_depth=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_bbox2d=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_consistency=dict(type='GIoULoss', loss_weight=1.0),
                 pred_bbox2d=True,
                 pred_keypoints=False,
                 bbox_coder=dict(
                     type='PGDBBoxCoder',
                     base_depths=((28.01, 16.32), ),
                     base_dims=((0.8, 1.73, 0.6), (1.76, 1.73, 0.6),
                                (3.9, 1.56, 1.6)),
                     code_size=7),
                 **kwargs):
        self.topk = 15
        self.covariance_type = 'diag'
        self.use_depth_classifier = use_depth_classifier
        self.use_onlyreg_proj = use_onlyreg_proj
        self.depth_branch = depth_branch
        self.pred_keypoints = pred_keypoints
        self.weight_dim = weight_dim
        self.weight_branch = weight_branch
        self.weight_out_channels = []
        for weight_branch_channels in weight_branch:
            if len(weight_branch_channels) > 0:
                self.weight_out_channels.append(weight_branch_channels[-1])
            else:
                self.weight_out_channels.append(-1)
        self.depth_range = depth_range
        self.depth_unit = depth_unit
        self.division = division
        if self.division == 'uniform':
            self.num_depth_cls = int(
                (depth_range[1] - depth_range[0]) / depth_unit) + 1
            if self.num_depth_cls != depth_bins:
                print('Warning: The number of bins computed from ' +
                      'depth_unit is different from given parameter! ' +
                      'Depth_unit will be considered with priority in ' +
                      'Uniform Division.')
        else:
            self.num_depth_cls = depth_bins
        super().__init__(
            pred_bbox2d=pred_bbox2d, bbox_coder=bbox_coder, **kwargs)
        self.loss_depth = build_loss(loss_depth)
        if self.pred_bbox2d:
            self.loss_bbox2d = build_loss(loss_bbox2d)
            self.loss_consistency = build_loss(loss_consistency)
        if self.pred_keypoints:
            self.kpts_start = 9 if self.pred_velo else 7

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        if self.pred_bbox2d:
            self.scale_dim += 1
        if self.pred_keypoints:
            self.scale_dim += 1
        self.scales = nn.ModuleList([
            nn.ModuleList([Scale(1.0) for _ in range(self.scale_dim)])
            for _ in self.strides
        ])

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        super()._init_predictor()

        if self.use_depth_classifier:
            self.conv_depth_cls_prev = self._init_branch(
                conv_channels=self.depth_branch,
                conv_strides=(1, ) * len(self.depth_branch))
            self.conv_depth_cls = nn.Conv2d(self.depth_branch[-1],
                                            self.num_depth_cls, 1)
            # Data-agnostic single param lambda for local depth fusion
            self.fuse_lambda = nn.Parameter(torch.tensor(10e-5))

        if self.weight_dim != -1:
            self.conv_weight_prevs = nn.ModuleList()
            self.conv_weights = nn.ModuleList()
            for i in range(self.weight_dim):
                weight_branch_channels = self.weight_branch[i]
                weight_out_channel = self.weight_out_channels[i]
                if len(weight_branch_channels) > 0:
                    self.conv_weight_prevs.append(
                        self._init_branch(
                            conv_channels=weight_branch_channels,
                            conv_strides=(1, ) * len(weight_branch_channels)))
                    self.conv_weights.append(
                        nn.Conv2d(weight_out_channel, 1, 1))
                else:
                    self.conv_weight_prevs.append(None)
                    self.conv_weights.append(
                        nn.Conv2d(self.feat_channels, 1, 1))

    def init_weights(self):
        """Initialize weights of the head.

        We currently still use the customized defined init_weights because the
        default init of DCN triggered by the init_cfg will init
        conv_offset.weight, which mistakenly affects the training stability.
        """
        super().init_weights()

        bias_cls = bias_init_with_prob(0.01)
        if self.use_depth_classifier:
            for m in self.conv_depth_cls_prev:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
            normal_init(self.conv_depth_cls, std=0.01, bias=bias_cls)

        if self.weight_dim != -1:
            for conv_weight_prev in self.conv_weight_prevs:
                if conv_weight_prev is None:
                    continue
                for m in conv_weight_prev:
                    if isinstance(m.conv, nn.Conv2d):
                        normal_init(m.conv, std=0.01)
            for conv_weight in self.conv_weights:
                normal_init(conv_weight, std=0.01)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level,
                    each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * bbox_code_size.
                dir_cls_preds (list[Tensor]): Box scores for direction class
                    predictions on each scale level, each is a 4D-tensor,
                    the channel number is num_points * 2. (bin = 2).
                weight (list[Tensor]): Location-aware weight maps on each
                    scale level, each is a 4D-tensor, the channel number is
                    num_points * 1.
                depth_cls_preds (list[Tensor]): Box scores for depth class
                    predictions on each scale level, each is a 4D-tensor,
                    the channel number is num_points * self.num_depth_cls.
                attr_preds (list[Tensor]): Attribute scores for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * num_attrs.
                centernesses (list[Tensor]): Centerness for each scale level,
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox and direction class
                predictions, depth class predictions, location-aware weights,
                attribute and centerness predictions of input feature maps.
        """
        cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness, cls_feat, \
            reg_feat = super().forward_single(x, scale, stride)

        max_regress_range = stride * self.regress_ranges[0][1] / \
            self.strides[0]
        bbox_pred = self.bbox_coder.decode_2d(bbox_pred, scale, stride,
                                              max_regress_range, self.training,
                                              self.pred_keypoints,
                                              self.pred_bbox2d)

        depth_cls_pred = None
        if self.use_depth_classifier:
            clone_reg_feat = reg_feat.clone()
            for conv_depth_cls_prev_layer in self.conv_depth_cls_prev:
                clone_reg_feat = conv_depth_cls_prev_layer(clone_reg_feat)
            depth_cls_pred = self.conv_depth_cls(clone_reg_feat)

        weight = None
        if self.weight_dim != -1:
            weight = []
            for i in range(self.weight_dim):
                clone_reg_feat = reg_feat.clone()
                if len(self.weight_branch[i]) > 0:
                    for conv_weight_prev_layer in self.conv_weight_prevs[i]:
                        clone_reg_feat = conv_weight_prev_layer(clone_reg_feat)
                weight.append(self.conv_weights[i](clone_reg_feat))
            weight = torch.cat(weight, dim=1)

        return cls_score, bbox_pred, dir_cls_pred, depth_cls_pred, weight, \
            attr_pred, centerness

    def get_proj_bbox2d(self,
                        bbox_preds,
                        pos_dir_cls_preds,
                        labels_3d,
                        bbox_targets_3d,
                        pos_points,
                        pos_inds,
                        img_metas,
                        pos_depth_cls_preds=None,
                        pos_weights=None,
                        pos_cls_scores=None,
                        with_kpts=False):
        """Decode box predictions and get projected 2D attributes.

        Args:
            bbox_preds (list[Tensor]): Box predictions for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * bbox_code_size.
            pos_dir_cls_preds (Tensor): Box scores for direction class
                predictions of positive boxes on all the scale levels in shape
                (num_pos_points, 2).
            labels_3d (list[Tensor]): 3D box category labels for each scale
                level, each is a 4D-tensor.
            bbox_targets_3d (list[Tensor]): 3D box targets for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * bbox_code_size.
            pos_points (Tensor): Foreground points.
            pos_inds (Tensor): Index of foreground points from flattened
                tensors.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            pos_depth_cls_preds (Tensor, optional): Probabilistic depth map of
                positive boxes on all the scale levels in shape
                (num_pos_points, self.num_depth_cls). Defaults to None.
            pos_weights (Tensor, optional): Location-aware weights of positive
                boxes in shape (num_pos_points, self.weight_dim). Defaults to
                None.
            pos_cls_scores (Tensor, optional): Classification scores of
                positive boxes in shape (num_pos_points, self.num_classes).
                Defaults to None.
            with_kpts (bool, optional): Whether to output keypoints targets.
                Defaults to False.

        Returns:
            tuple[Tensor]: Exterior 2D boxes from projected 3D boxes,
                predicted 2D boxes and keypoint targets (if necessary).
        """
        views = [np.array(img_meta['cam2img']) for img_meta in img_metas]
        num_imgs = len(img_metas)
        img_idx = []
        for label in labels_3d:
            for idx in range(num_imgs):
                img_idx.append(
                    labels_3d[0].new_ones(int(len(label) / num_imgs)) * idx)
        img_idx = torch.cat(img_idx)
        pos_img_idx = img_idx[pos_inds]

        flatten_strided_bbox_preds = []
        flatten_strided_bbox2d_preds = []
        flatten_bbox_targets_3d = []
        flatten_strides = []

        for stride_idx, bbox_pred in enumerate(bbox_preds):
            flatten_bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(
                -1, sum(self.group_reg_dims))
            flatten_bbox_pred[:, :2] *= self.strides[stride_idx]
            flatten_bbox_pred[:, -4:] *= self.strides[stride_idx]
            flatten_strided_bbox_preds.append(
                flatten_bbox_pred[:, :self.bbox_coder.bbox_code_size])
            flatten_strided_bbox2d_preds.append(flatten_bbox_pred[:, -4:])

            bbox_target_3d = bbox_targets_3d[stride_idx].clone()
            bbox_target_3d[:, :2] *= self.strides[stride_idx]
            bbox_target_3d[:, -4:] *= self.strides[stride_idx]
            flatten_bbox_targets_3d.append(bbox_target_3d)

            flatten_stride = flatten_bbox_pred.new_ones(
                *flatten_bbox_pred.shape[:-1], 1) * self.strides[stride_idx]
            flatten_strides.append(flatten_stride)

        flatten_strided_bbox_preds = torch.cat(flatten_strided_bbox_preds)
        flatten_strided_bbox2d_preds = torch.cat(flatten_strided_bbox2d_preds)
        flatten_bbox_targets_3d = torch.cat(flatten_bbox_targets_3d)
        flatten_strides = torch.cat(flatten_strides)
        pos_strided_bbox_preds = flatten_strided_bbox_preds[pos_inds]
        pos_strided_bbox2d_preds = flatten_strided_bbox2d_preds[pos_inds]
        pos_bbox_targets_3d = flatten_bbox_targets_3d[pos_inds]
        pos_strides = flatten_strides[pos_inds]

        pos_decoded_bbox2d_preds = distance2bbox(pos_points,
                                                 pos_strided_bbox2d_preds)

        pos_strided_bbox_preds[:, :2] = \
            pos_points - pos_strided_bbox_preds[:, :2]
        pos_bbox_targets_3d[:, :2] = \
            pos_points - pos_bbox_targets_3d[:, :2]

        if self.use_depth_classifier and (not self.use_onlyreg_proj):
            pos_prob_depth_preds = self.bbox_coder.decode_prob_depth(
                pos_depth_cls_preds, self.depth_range, self.depth_unit,
                self.division, self.num_depth_cls)
            sig_alpha = torch.sigmoid(self.fuse_lambda)
            pos_strided_bbox_preds[:, 2] = \
                sig_alpha * pos_strided_bbox_preds.clone()[:, 2] + \
                (1 - sig_alpha) * pos_prob_depth_preds

        box_corners_in_image = pos_strided_bbox_preds.new_zeros(
            (*pos_strided_bbox_preds.shape[:-1], 8, 2))
        box_corners_in_image_gt = pos_strided_bbox_preds.new_zeros(
            (*pos_strided_bbox_preds.shape[:-1], 8, 2))

        for idx in range(num_imgs):
            mask = (pos_img_idx == idx)
            if pos_strided_bbox_preds[mask].shape[0] == 0:
                continue
            cam2img = torch.eye(
                4,
                dtype=pos_strided_bbox_preds.dtype,
                device=pos_strided_bbox_preds.device)
            view_shape = views[idx].shape
            cam2img[:view_shape[0], :view_shape[1]] = \
                pos_strided_bbox_preds.new_tensor(views[idx])

            centers2d_preds = pos_strided_bbox_preds.clone()[mask, :2]
            centers2d_targets = pos_bbox_targets_3d.clone()[mask, :2]
            centers3d_targets = points_img2cam(pos_bbox_targets_3d[mask, :3],
                                               views[idx])

            # use predicted depth to re-project the 2.5D centers
            pos_strided_bbox_preds[mask, :3] = points_img2cam(
                pos_strided_bbox_preds[mask, :3], views[idx])
            pos_bbox_targets_3d[mask, :3] = centers3d_targets

            # depth fixed when computing re-project 3D bboxes
            pos_strided_bbox_preds[mask, 2] = \
                pos_bbox_targets_3d.clone()[mask, 2]

            # decode yaws
            if self.use_direction_classifier:
                pos_dir_cls_scores = torch.max(
                    pos_dir_cls_preds[mask], dim=-1)[1]
                pos_strided_bbox_preds[mask] = self.bbox_coder.decode_yaw(
                    pos_strided_bbox_preds[mask], centers2d_preds,
                    pos_dir_cls_scores, self.dir_offset, cam2img)
            pos_bbox_targets_3d[mask, 6] = torch.atan2(
                centers2d_targets[:, 0] - cam2img[0, 2],
                cam2img[0, 0]) + pos_bbox_targets_3d[mask, 6]

            corners = img_metas[0]['box_type_3d'](
                pos_strided_bbox_preds[mask],
                box_dim=self.bbox_coder.bbox_code_size,
                origin=(0.5, 0.5, 0.5)).corners
            box_corners_in_image[mask] = points_cam2img(corners, cam2img)

            corners_gt = img_metas[0]['box_type_3d'](
                pos_bbox_targets_3d[mask, :self.bbox_code_size],
                box_dim=self.bbox_coder.bbox_code_size,
                origin=(0.5, 0.5, 0.5)).corners
            box_corners_in_image_gt[mask] = points_cam2img(corners_gt, cam2img)

        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        proj_bbox2d_preds = torch.cat([minxy, maxxy], dim=1)

        outputs = (proj_bbox2d_preds, pos_decoded_bbox2d_preds)

        if with_kpts:
            norm_strides = pos_strides * self.regress_ranges[0][1] / \
                self.strides[0]
            kpts_targets = box_corners_in_image_gt - pos_points[..., None, :]
            kpts_targets = kpts_targets.view(
                (*pos_strided_bbox_preds.shape[:-1], 16))
            kpts_targets /= norm_strides

            outputs += (kpts_targets, )

        return outputs

    def get_pos_predictions(self, bbox_preds, dir_cls_preds, depth_cls_preds,
                            weights, attr_preds, centernesses, pos_inds,
                            img_metas):
        """Flatten predictions and get positive ones.

        Args:
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * bbox_code_size.
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * 2. (bin = 2)
            depth_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * self.num_depth_cls.
            attr_preds (list[Tensor]): Attribute scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_attrs.
            centernesses (list[Tensor]): Centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            pos_inds (Tensor): Index of foreground points from flattened
                tensors.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple[Tensor]: Box predictions, direction classes, probabilistic
                depth maps, location-aware weight maps, attributes and
                centerness predictions.
        """
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, sum(self.group_reg_dims))
            for bbox_pred in bbox_preds
        ]
        flatten_dir_cls_preds = [
            dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)
            for dir_cls_pred in dir_cls_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_dir_cls_preds = torch.cat(flatten_dir_cls_preds)
        flatten_centerness = torch.cat(flatten_centerness)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_dir_cls_preds = flatten_dir_cls_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]

        pos_depth_cls_preds = None
        if self.use_depth_classifier:
            flatten_depth_cls_preds = [
                depth_cls_pred.permute(0, 2, 3,
                                       1).reshape(-1, self.num_depth_cls)
                for depth_cls_pred in depth_cls_preds
            ]
            flatten_depth_cls_preds = torch.cat(flatten_depth_cls_preds)
            pos_depth_cls_preds = flatten_depth_cls_preds[pos_inds]

        pos_weights = None
        if self.weight_dim != -1:
            flatten_weights = [
                weight.permute(0, 2, 3, 1).reshape(-1, self.weight_dim)
                for weight in weights
            ]
            flatten_weights = torch.cat(flatten_weights)
            pos_weights = flatten_weights[pos_inds]

        pos_attr_preds = None
        if self.pred_attrs:
            flatten_attr_preds = [
                attr_pred.permute(0, 2, 3, 1).reshape(-1, self.num_attrs)
                for attr_pred in attr_preds
            ]
            flatten_attr_preds = torch.cat(flatten_attr_preds)
            pos_attr_preds = flatten_attr_preds[pos_inds]

        return pos_bbox_preds, pos_dir_cls_preds, pos_depth_cls_preds, \
            pos_weights, pos_attr_preds, pos_centerness

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'dir_cls_preds',
                  'depth_cls_preds', 'weights', 'attr_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             dir_cls_preds,
             depth_cls_preds,
             weights,
             attr_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             gt_bboxes_3d,
             gt_labels_3d,
             centers2d,
             depths,
             attr_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * bbox_code_size.
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * 2. (bin = 2)
            depth_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * self.num_depth_cls.
            weights (list[Tensor]): Location-aware weights for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * self.weight_dim.
            attr_preds (list[Tensor]): Attribute scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_attrs.
            centernesses (list[Tensor]): Centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_3d (list[Tensor]): 3D boxes ground truth with shape of
                (num_gts, code_size).
            gt_labels_3d (list[Tensor]): same as gt_labels
            centers2d (list[Tensor]): 2D centers on the image with shape of
                (num_gts, 2).
            depths (list[Tensor]): Depth ground truth with shape of
                (num_gts, ).
            attr_labels (list[Tensor]): Attributes indices of each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor]): specify which bounding boxes can
                be ignored when computing the loss. Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(dir_cls_preds) == \
            len(depth_cls_preds) == len(weights) == len(centernesses) == \
            len(attr_preds), 'The length of cls_scores, bbox_preds, ' \
            'dir_cls_preds, depth_cls_preds, weights, centernesses, and' \
            f'attr_preds: {len(cls_scores)}, {len(bbox_preds)}, ' \
            f'{len(dir_cls_preds)}, {len(depth_cls_preds)}, {len(weights)}' \
            f'{len(centernesses)}, {len(attr_preds)} are inconsistent.'
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)

        all_level_points_new = []
        for i in range (cls_scores[0].size(0)):
            all_level_points_new.append(all_level_points)

        # todo 获取的3D训练目标
        labels_3d, bbox_targets_3d, centerness_targets, attr_targets, \
        labels_3d2, bbox_targets_3d2, centerness_targets2, gt_inds = \
            self.get_targets(
                all_level_points, gt_bboxes, gt_labels, gt_bboxes_3d,
                gt_labels_3d, centers2d, depths, attr_labels)

        num_imgs = cls_scores[0].size(0)
        # todo 展平所有的预测结果
        # flatten cls_scores and targets
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_labels_3d = torch.cat(labels_3d)
        flatten_bbox_targets_3d = torch.cat(bbox_targets_3d)
        flatten_centerness_targets = torch.cat(centerness_targets)
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])
        if self.pred_attrs:
            flatten_attr_targets = torch.cat(attr_targets)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        # todo 给背景赋予类别，以及获取正样本的序号
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels_3d >= 0)
                    & (flatten_labels_3d < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)

        # todo 采用GA重新分配的样本的
        # todo 1、计算每个样本的损失
        # todo flatten_cls_scores = [nums, 3], flatten_labels_3d =[nums, 1]
        # todo flatten_bbox_targets_3d = [nums, 11], flatten_points = [nums, 2]

        # todo 首先需要将所有的转换到一张图片上
        image_cls_scores = levels_to_images(cls_scores)
        image_bbox_preds = levels_to_images(bbox_preds)

        # todo 为每一张图片生成属于自己的正样本
        image_pos_inds = []
        for i in range(len(labels_3d2)):
            ss = ((labels_3d2[i] >= 0)
                    & (labels_3d2[i] < bg_class_ind)).nonzero().reshape(-1)
            image_pos_inds.append(ss)

        # todo 计算每个样本的损失
        pos_loss_list, pll_all, lc, lr = multi_apply(self.get_pos_loss, image_cls_scores,
                                                     image_bbox_preds, labels_3d2,
                                                     bbox_targets_3d2, image_pos_inds, centerness_targets2)

        # todo 2、根据损失重新分配
        with torch.no_grad():
            reassign_labels, reassign_label_weight, \
            num_pos_new = multi_apply(
                self.ga_reassign,
                pos_loss_list,
                labels_3d2,
                image_pos_inds,
                gt_inds,
                all_level_points_new,
                pll_all,
                lc, lr)
            num_pos_new = sum(num_pos_new)

        # todo 重新去获取正样本的参数
        # todo 重新排列样本的序号状态，[all_l1, all_l2, all_l3, all_l4]
        # the number of points per img, per lvl
        num_points = [center.size(0) for center in all_level_points]
        reassign_labels = [
            mm.split(num_points, 0) for mm in reassign_labels
        ]

        con_reassign_labels =[]
        num_levels = len(all_level_points)
        for i in  range(num_levels):
            con_reassign_labels.append(
                torch.cat([nn[i] for nn in reassign_labels]))
        # for i in range(len(reassign_labels)):
        #     for j in range(len(reassign_labels[i])):
        #         print(reassign_labels[i][j].shape)
        # for i in range(len(con_reassign_labels)):
        #     print(con_reassign_labels[i].shape)
        # exit()

        # todo 将正样本展平
        flatten_label_news = torch.cat(con_reassign_labels, 0).view(-1)
        pos_inds_new = ((flatten_label_news >= 0)
                    & (flatten_label_news < bg_class_ind)).nonzero().reshape(-1)

        flatten_labels_3d = flatten_label_news
        pos_inds = pos_inds_new
        num_pos = num_pos_new

        # todo 计算的正样本的回归的参数
        pos_bbox_preds, pos_dir_cls_preds, pos_depth_cls_preds, pos_weights, \
            pos_attr_preds, pos_centerness = self.get_pos_predictions(
                bbox_preds, dir_cls_preds, depth_cls_preds, weights,
                attr_preds, centernesses, pos_inds, img_metas)
        # todo 计算损失
        loss_dict = dict()
        loss_dict['loss_cls'] = self.loss_cls(
            flatten_cls_scores,
            flatten_labels_3d,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        if num_pos > 0:
            # todo 正样本
            pos_bbox_targets_3d = flatten_bbox_targets_3d[pos_inds]
            pos_centerness_targets = flatten_centerness_targets[pos_inds]
            pos_points = flatten_points[pos_inds]
            # todo 预测辅助
            if self.pred_attrs:
                pos_attr_targets = flatten_attr_targets[pos_inds]
            # todo 使用方向分类
            if self.use_direction_classifier:
                pos_dir_cls_targets = self.get_direction_target(
                    pos_bbox_targets_3d, self.dir_offset, one_hot=False)

            # todo 回归的权重
            bbox_weights = pos_centerness_targets.new_ones(
                len(pos_centerness_targets), sum(self.group_reg_dims))
            equal_weights = pos_centerness_targets.new_ones(
                pos_centerness_targets.shape)
            code_weight = self.train_cfg.get('code_weight', None)
            if code_weight:
                assert len(code_weight) == sum(self.group_reg_dims)
                bbox_weights = bbox_weights * bbox_weights.new_tensor(
                    code_weight)

            # todo 角度的正弦损失
            if self.diff_rad_by_sin:
                pos_bbox_preds, pos_bbox_targets_3d = self.add_sin_difference(
                    pos_bbox_preds, pos_bbox_targets_3d)
            # todo 位置偏移损失
            loss_dict['loss_offset'] = self.loss_bbox(
                pos_bbox_preds[:, :2],
                pos_bbox_targets_3d[:, :2],
                weight=bbox_weights[:, :2],
                avg_factor=equal_weights.sum())
            # todo 长宽高
            loss_dict['loss_size'] = self.loss_bbox(
                pos_bbox_preds[:, 3:6],
                pos_bbox_targets_3d[:, 3:6],
                weight=bbox_weights[:, 3:6],
                avg_factor=equal_weights.sum())
            # todo 旋转的角度
            loss_dict['loss_rotsin'] = self.loss_bbox(
                pos_bbox_preds[:, 6],
                pos_bbox_targets_3d[:, 6],
                weight=bbox_weights[:, 6],
                avg_factor=equal_weights.sum())
            # todo 速度的损失
            if self.pred_velo:
                loss_dict['loss_velo'] = self.loss_bbox(
                    pos_bbox_preds[:, 7:9],
                    pos_bbox_targets_3d[:, 7:9],
                    weight=bbox_weights[:, 7:9],
                    avg_factor=equal_weights.sum())

            proj_bbox2d_inputs = (bbox_preds, pos_dir_cls_preds, labels_3d,
                                  bbox_targets_3d, pos_points, pos_inds,
                                  img_metas)

            # direction classification loss
            # TODO: add more check for use_direction_classifier
            if self.use_direction_classifier:
                loss_dict['loss_dir'] = self.loss_dir(
                    pos_dir_cls_preds,
                    pos_dir_cls_targets,
                    equal_weights,
                    avg_factor=equal_weights.sum())

            # init depth loss with the one computed from direct regression
            loss_dict['loss_depth'] = self.loss_bbox(
                pos_bbox_preds[:, 2],
                pos_bbox_targets_3d[:, 2],
                weight=bbox_weights[:, 2],
                avg_factor=equal_weights.sum())
            # depth classification loss
            if self.use_depth_classifier:
                pos_prob_depth_preds = self.bbox_coder.decode_prob_depth(
                    pos_depth_cls_preds, self.depth_range, self.depth_unit,
                    self.division, self.num_depth_cls)
                sig_alpha = torch.sigmoid(self.fuse_lambda)
                if self.weight_dim != -1:
                    loss_fuse_depth = self.loss_depth(
                        sig_alpha * pos_bbox_preds[:, 2] +
                        (1 - sig_alpha) * pos_prob_depth_preds,
                        pos_bbox_targets_3d[:, 2],
                        sigma=pos_weights[:, 0],
                        weight=bbox_weights[:, 2],
                        avg_factor=equal_weights.sum())
                else:
                    loss_fuse_depth = self.loss_depth(
                        sig_alpha * pos_bbox_preds[:, 2] +
                        (1 - sig_alpha) * pos_prob_depth_preds,
                        pos_bbox_targets_3d[:, 2],
                        weight=bbox_weights[:, 2],
                        avg_factor=equal_weights.sum())
                loss_dict['loss_depth'] = loss_fuse_depth

                proj_bbox2d_inputs += (pos_depth_cls_preds, )

            if self.pred_keypoints:
                # use smoothL1 to compute consistency loss for keypoints
                # normalize the offsets with strides
                proj_bbox2d_preds, pos_decoded_bbox2d_preds, kpts_targets = \
                    self.get_proj_bbox2d(*proj_bbox2d_inputs, with_kpts=True)
                loss_dict['loss_kpts'] = self.loss_bbox(
                    pos_bbox_preds[:, self.kpts_start:self.kpts_start + 16],
                    kpts_targets,
                    weight=bbox_weights[:,
                                        self.kpts_start:self.kpts_start + 16],
                    avg_factor=equal_weights.sum())

            if self.pred_bbox2d:
                loss_dict['loss_bbox2d'] = self.loss_bbox2d(
                    pos_bbox_preds[:, -4:],
                    pos_bbox_targets_3d[:, -4:],
                    weight=bbox_weights[:, -4:],
                    avg_factor=equal_weights.sum())
                if not self.pred_keypoints:
                    proj_bbox2d_preds, pos_decoded_bbox2d_preds = \
                        self.get_proj_bbox2d(*proj_bbox2d_inputs)
                loss_dict['loss_consistency'] = self.loss_consistency(
                    proj_bbox2d_preds,
                    pos_decoded_bbox2d_preds,
                    weight=bbox_weights[:, -4:],
                    avg_factor=equal_weights.sum())

            loss_dict['loss_centerness'] = self.loss_centerness(
                pos_centerness, pos_centerness_targets)

            # attribute classification loss
            if self.pred_attrs:
                loss_dict['loss_attr'] = self.loss_attr(
                    pos_attr_preds,
                    pos_attr_targets,
                    pos_centerness_targets,
                    avg_factor=pos_centerness_targets.sum())

        else:
            # need absolute due to possible negative delta x/y
            loss_dict['loss_offset'] = pos_bbox_preds[:, :2].sum()
            loss_dict['loss_size'] = pos_bbox_preds[:, 3:6].sum()
            loss_dict['loss_rotsin'] = pos_bbox_preds[:, 6].sum()
            loss_dict['loss_depth'] = pos_bbox_preds[:, 2].sum()
            if self.pred_velo:
                loss_dict['loss_velo'] = pos_bbox_preds[:, 7:9].sum()
            if self.pred_keypoints:
                loss_dict['loss_kpts'] = pos_bbox_preds[:,
                                                        self.kpts_start:self.
                                                        kpts_start + 16].sum()
            if self.pred_bbox2d:
                loss_dict['loss_bbox2d'] = pos_bbox_preds[:, -4:].sum()
                loss_dict['loss_consistency'] = pos_bbox_preds[:, -4:].sum()
            loss_dict['loss_centerness'] = pos_centerness.sum()
            if self.use_direction_classifier:
                loss_dict['loss_dir'] = pos_dir_cls_preds.sum()
            if self.use_depth_classifier:
                sig_alpha = torch.sigmoid(self.fuse_lambda)
                if self.weight_dim != -1:
                    loss_fuse_depth *= torch.exp(-pos_weights[:, 0].sum())
                else:
                    loss_fuse_depth = \
                        sig_alpha * pos_bbox_preds[:, 2].sum() + \
                        (1 - sig_alpha) * pos_depth_cls_preds.sum()
                loss_dict['loss_depth'] = loss_fuse_depth
            if self.pred_attrs:
                loss_dict['loss_attr'] = pos_attr_preds.sum()
        return loss_dict

    def get_pos_loss(self, cls_scores, bbox_pred,
                    labels_3d, bbox_targets_3d,
                     image_pos_inds,centerness_targets):
        if not len(image_pos_inds):
            return cls_scores.new([]),
        pos_scores = cls_scores[image_pos_inds]
        pos_bbox_pred = bbox_pred[image_pos_inds]
        pos_label = labels_3d[image_pos_inds]
        pos_target = bbox_targets_3d[image_pos_inds]
        pos_center = centerness_targets[image_pos_inds]
        # todo 解决一下的权重尺度的对应的问题 ！！！！！！！！！！！
        pos_weight = torch.ones(pos_target.shape).cuda() * torch.unsqueeze(pos_center, 1)
        pos_nums = len(image_pos_inds)

        if self.diff_rad_by_sin:
            pos_bbox_pred, pos_target = self.add_sin_difference(
                pos_bbox_pred, pos_target)

        loss_cls = self.loss_cls(pos_scores, pos_label, avg_factor=pos_nums + 1, reduction_override='none')
        loss_cls = loss_cls.sum(-1)

        loss_off_set = self.loss_bbox(pos_bbox_pred[:, :2],
                                      pos_target[:, :2],
                                      weight=pos_weight[:, :2],
                                      avg_factor = pos_center.sum(), reduction_override='none')
        loss_off_set = loss_off_set.sum(-1)

        loss_size = self.loss_bbox(pos_bbox_pred[:, 3:6],
                                      pos_target[:, 3:6],
                                      weight=pos_weight[:, 3:6],
                                      avg_factor = pos_center.sum(), reduction_override='none')
        loss_size = loss_size.sum(-1)

        loss_rotsin = self.loss_bbox(pos_bbox_pred[:, 6],
                                      pos_target[:, 6],
                                      weight=pos_weight[:, 6],
                                      avg_factor = pos_center.sum(), reduction_override='none')

        loss_bbox = loss_off_set + loss_size + loss_rotsin
        poss_loss = loss_bbox + loss_cls
        poss_all = torch.stack((loss_cls, loss_bbox), 1)
        return poss_loss, poss_all, loss_cls, loss_bbox

    # todo 采用优化算法***************************************************
    def ga_reassign(self, pos_losses, label, pos_inds, pos_gt_inds, anchors, pll_all, lc, lr):
        """Fit loss to GMM distribution and separate positive, ignore, negative
        samples again with GMM model.

        Args:
            pos_losses (Tensor): Losses of all positive samples in
                single image.
            label (Tensor): classification target of each anchor with
                shape (num_anchors,)
            label_weight (Tensor): Classification loss weight of each
                anchor with shape (num_anchors).
            bbox_weight (Tensor): Bbox weight of each anchor with shape
                (num_anchors, 4).
            pos_inds (Tensor): Index of all positive samples got from
                first assign process.
            pos_gt_inds (Tensor): Gt_index of all positive samples got
                from first assign process.
            anchors (list[Tensor]): Anchors of each scale.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - label (Tensor): classification target of each anchor after
                  paa assign, with shape (num_anchors,)
                - label_weight (Tensor): Classification loss weight of each
                  anchor after paa assign, with shape (num_anchors).
                - bbox_weight (Tensor): Bbox weight of each anchor with shape
                  (num_anchors, 4).
                - num_pos (int): The number of positive samples after paa
                  assign.
        """
        label_weight = torch.ones(label.shape).cuda()
        if not len(pos_inds):
            return label, label_weight, 0
        pos_gt_inds = pos_gt_inds[pos_inds]
        label = label.clone()
        label_weight = label_weight.clone()
        num_gt = pos_gt_inds.max() + 1
        num_level = len(anchors)
        num_anchors_each_level = [item.size(0) for item in anchors]
        num_anchors_each_level.insert(0, 0)
        inds_level_interval = np.cumsum(num_anchors_each_level)
        pos_level_mask = []
        # todo 分层构建的 mask 筛选出的每一层上的真值
        for i in range(num_level):
            mask = (pos_inds >= inds_level_interval[i]) & (
                    pos_inds < inds_level_interval[i + 1])
            pos_level_mask.append(mask)
        # todo pos_level_mask 用来指示每一层属于这一层上的物体 [pos, 5]
        pos_inds_after_paa = [label.new_tensor([])]
        ignore_inds_after_paa = [label.new_tensor([])]
        # todo 为每一个 gt 计算属于自己的正样本。 每个的gt的多个样本被用来的分离
        for gt_ind in range(num_gt):
            pos_inds_gmm = []
            pos_loss_gmm = []
            # todo 自己设计的储存列表
            pos_inds_svm = []
            pos_loss_svm = []
            pos_lc_svm = []
            pos_lr_svm = []
            gt_mask = pos_gt_inds == gt_ind
            for level in range(num_level):
                level_mask = pos_level_mask[level]
                # todo 过滤出属于每一个真值的每一个层级的正样本
                level_gt_mask = level_mask & gt_mask
                print(level_gt_mask.sum())
                # todo 这个地方应该是每个层级挑选了前9个样本（最少的为属于的它的正样本）
                value, topk_inds = pos_losses[level_gt_mask].topk(
                    min(level_gt_mask.sum(), self.topk), largest=False)
                pos_inds_gmm.append(pos_inds[level_gt_mask][topk_inds])
                pos_loss_gmm.append(value)
                # todo 设计自己的二元列表
                pos_inds_svm.append(pos_inds[level_gt_mask][topk_inds])
                pos_loss_svm.append(pll_all[level_gt_mask][topk_inds])
                pos_lc_svm.append(lc[level_gt_mask][topk_inds])
                pos_lr_svm.append(lr[level_gt_mask][topk_inds])
            # todo 对于的一个gt来说最多只有45个样本
            # todo 拼接在一起计算
            pos_inds_gmm = torch.cat(pos_inds_gmm)
            pos_loss_gmm = torch.cat(pos_loss_gmm)
            # pos_loss_svm = torch.cat(pos_loss_svm)
            pos_inds_svm = torch.cat(pos_inds_svm)
            pos_lc_svm = torch.cat(pos_lc_svm)
            pos_lr_svm = torch.cat(pos_lr_svm)
            # todo 看一下的可视化的结果
            x1 = pos_lc_svm.detach().cpu().numpy()
            y1 = pos_lr_svm.detach().cpu().numpy()

            # plt.scatter(x1, y1)
            # plt.savefig('results1.jpg')

            """ 最优化算法 """
            # todo 此处使用了最优化算法进行计算************************************************
            # mean_x = x1.mean()
            # mean_y = y1.mean()
            # b_init = 1
            # c_init = -(mean_x + mean_y)
            #
            # # todo 定义一个最优化的目标
            # def disloss(p):
            #     b, c = p
            #     dis = (np.abs(x1 + b * y1 + c) / np.sqrt(1 + b ** 2)).sum(0)
            #     return dis
            #
            # # todo 调用优化算法求解的
            # # todo Nelder-Mead BFGS TNC CG Powell
            # mymin = minimize(disloss, (b_init, c_init), method='CG')
            # # todo 优化后的最小值,以及解析解
            # outs = mymin.fun
            # results = mymin.x
            # todo 此处使用了最优化算法进行计算************************************************
            """ 最优化算法 """

            """ 遗传学算法 """
            # todo 此处使用了遗传学算法进行计算************************************************
            class myproblem(ea.Problem):
                def __init__(self):
                    name = 'MyProblem'
                    M = 1 # todo 初始化Dim
                    maxormins = [1] # todo 1:最小化， -1最大化
                    Dim = 2 # todo 决策变量维度，
                    varTypes = [0] * Dim
                    lb = [1, -10] # todo 决策变量下界
                    ub = [10, 0] # todo 决策变量上界
                    lbin = [0, 1] # todo 决策变量下边界 0不包含
                    ubin = [1, 0] # todo 决策变量下边界 1包含
                    ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

                def aimFunc(self, pop):
                    Vars = pop.Phen
                    b = Vars[:, [0]]
                    c = Vars[:, [1]]
                    pop.ObjV = (np.abs(x1 + b * y1 + c) / np.sqrt(1 + b ** 2)).sum(axis=1, keepdims=True)

            problem = myproblem()
            Encoding = 'RI' # todo 编码方式
            NIND = 150 # todo 种群规模
            Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
            population = ea.Population(Encoding, Field, NIND)

            # myalgorathm = ea.soea_SEGA_templet(problem, population)
            myalgorathm = ea.soea_DE_rand_1_bin_templet(problem, population)
            myalgorathm.MAXGEN = 100 # todo 最大进化代数
            myalgorathm.mutOper.F = 0.5 # todo 差分进化中的参数F
            myalgorathm.recOper.XOVR = 0.7 # todo 重组概率
            myalgorathm.logTras = 0
            myalgorathm.verbose = False
            myalgorathm.drawing = 0

            [BestIndi, population] = myalgorathm.run()
            outs = BestIndi.ObjV[0][0]
            results = BestIndi.Phen[0]
            # todo 此处使用了遗传学算法进行计算************************************************
            """ 遗传学算法 """

            # todo 为所有的点计算到经过原点的但是的斜率和分界线的斜率相同的直线
            b1 = results[0]
            c1 = 0
            distance = np.abs(x1 + b1 * y1 + c1)/np.sqrt(1 + b1 **2)
            # todo 计算的两条直线的距离
            line_dis = np.abs(results[1]) / np.sqrt(1 + b1 ** 2)
            # todo 边界判断， 处于两条平行线中间的是正样本的，否则就为负样本。
            cate = (distance > line_dis) + 0

            # todo 修复ga
            if len(pos_inds_svm) < 2:
                continue
            # todo 重新分配标签
            device = pos_inds_svm.device
            cate = torch.from_numpy(cate).to(device)
            ga_pos_inds_temp, ga_ignore_inds_temp = self.ga_separation_scheme(
                cate, pos_inds_svm)
            pos_inds_after_paa.append(ga_pos_inds_temp)
            ignore_inds_after_paa.append(ga_ignore_inds_temp)

            """ paa or"""
            # todo *****************************************************
            # todo 修复gmm
            # fix gmm need at least two sample
            # if len(pos_inds_gmm) < 2:
            #     continue
            #
            # # todo 此处获得了 pos_loss_gmm 重新按照顺序进行排序, 将损失的按照升序进行排序
            # device = pos_inds_gmm.device
            # pos_loss_gmm, sort_inds = pos_loss_gmm.sort()
            # pos_inds_gmm = pos_inds_gmm[sort_inds]
            # pos_loss_gmm = pos_loss_gmm.view(-1, 1).cpu().numpy()
            # # # todo 初始化数据准备
            # min_loss, max_loss = pos_loss_gmm.min(), pos_loss_gmm.max()
            # means_init = np.array([min_loss, max_loss]).reshape(2, 1)
            # weights_init = np.array([0.5, 0.5])
            # precisions_init = np.array([1.0, 1.0]).reshape(2, 1, 1)  # full
            # if self.covariance_type == 'spherical':
            #     precisions_init = precisions_init.reshape(2)
            # elif self.covariance_type == 'diag':
            #     precisions_init = precisions_init.reshape(2, 1)
            # elif self.covariance_type == 'tied':
            #     precisions_init = np.array([[1.0]])
            # if skm is None:
            #     raise ImportError('Please run "pip install sklearn" '
            #                       'to install sklearn first.')
            # gmm = skm.GaussianMixture(
            #     2,
            #     weights_init=weights_init,
            #     means_init=means_init,
            #     precisions_init=precisions_init,
            #     covariance_type=self.covariance_type,
            #     reg_covar=1e-4)
            # gmm.fit(pos_loss_gmm)
            # gmm_assignment = gmm.predict(pos_loss_gmm)
            # scores = gmm.score_samples(pos_loss_gmm)
            # gmm_assignment = torch.from_numpy(gmm_assignment).to(device)
            # scores = torch.from_numpy(scores).to(device)
            #
            # pos_inds_temp, ignore_inds_temp = self.gmm_separation_scheme(
            #     gmm_assignment, scores, pos_inds_gmm)
            # pos_inds_after_paa.append(pos_inds_temp)
            # ignore_inds_after_paa.append(ignore_inds_temp)
            # todo *****************************************************
            """ paa or"""

        pos_inds_after_paa = torch.cat(pos_inds_after_paa)
        ignore_inds_after_paa = torch.cat(ignore_inds_after_paa)
        reassign_mask = (pos_inds.unsqueeze(1) != pos_inds_after_paa).all(1)
        reassign_ids = pos_inds[reassign_mask]
        # todo 先把所有的label全部置为10
        label[reassign_ids] = self.background_label
        # todo 被忽略的样本的权重设为0， 即不参与训练
        label_weight[ignore_inds_after_paa] = 0
        # todo 将重新分配后的与之前的进行对比，凡是没有被分配的pos，这次权重全部重新置0
        num_pos = len(pos_inds_after_paa)
        return label, label_weight, num_pos

    def ga_separation_scheme(self, ga_assignment, pos_inds_gmm):
        # todo gmm_ass = [num] scores = [num]
        # todo 首先把assignment=0 的找出来 fgs应该是前景, 最多是45个
        fgs = ga_assignment == 0
        # todo 设置等待分配的正样本
        pos_inds_temp = fgs.new_tensor([], dtype=torch.long)
        ignore_inds_temp = fgs.new_tensor([], dtype=torch.long)
        # todo numel() 返回一个矩阵中元素的个数 nonzero()返回的是数组中非零元素的位置
        # todo 当有正样本的时候进行分配
        if fgs.nonzero().numel():
            # todo 只要fgs存在东西就是正样本
            pos_inds_temp = pos_inds_gmm[fgs]
            # _, pos_thr_ind = scores[fgs].topk(1)
            # # todo 将前景的列表设置 ，pos_inds_gmm:正样本的编号， 先从正样本的中筛选出前景， 再从所有的前景中筛选出从零到阈值的样本
            # # todo 由于是按照损失的排序，损失小的都在前面，即为正样本。
            # pos_inds_temp = pos_inds_gmm[fgs][:pos_thr_ind + 1]
            # todo 将忽略的样本的列表设置为空
            ignore_inds_temp = pos_inds_gmm.new_tensor([])
        return pos_inds_temp, ignore_inds_temp

    def gmm_separation_scheme(self, gmm_assignment, scores, pos_inds_gmm):
        """A general separation scheme for gmm model.

        It separates a GMM distribution of candidate samples into three
        parts, 0 1 and uncertain areas, and you can implement other
        separation schemes by rewriting this function.

        Args:
            gmm_assignment (Tensor): The prediction of GMM which is of shape
                (num_samples,). The 0/1 value indicates the distribution
                that each sample comes from.
            scores (Tensor): The probability of sample coming from the
                fit GMM distribution. The tensor is of shape (num_samples,).
            pos_inds_gmm (Tensor): All the indexes of samples which are used
                to fit GMM model. The tensor is of shape (num_samples,)

        Returns:
            tuple[Tensor]: The indices of positive and ignored samples.

                - pos_inds_temp (Tensor): Indices of positive samples.
                - ignore_inds_temp (Tensor): Indices of ignore samples.
        """
        # The implementation is (c) in Fig.3 in origin paper instead of (b).
        # todo 这个地方说明，没有被忽略的样本， 样本被分为正样本和负样本
        # You can refer to issues such as
        # https://github.com/kkhoot/PAA/issues/8 and
        # https://github.com/kkhoot/PAA/issues/9.
        fgs = gmm_assignment == 0
        pos_inds_temp = fgs.new_tensor([], dtype=torch.long)
        ignore_inds_temp = fgs.new_tensor([], dtype=torch.long)
        if fgs.nonzero().numel():
            _, pos_thr_ind = scores[fgs].topk(1)
            pos_inds_temp = pos_inds_gmm[fgs][:pos_thr_ind + 1]
            ignore_inds_temp = pos_inds_gmm.new_tensor([])
        return pos_inds_temp, ignore_inds_temp
    # todo ************************************************************

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'dir_cls_preds',
                  'depth_cls_preds', 'weights', 'attr_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   dir_cls_preds,
                   depth_cls_preds,
                   weights,
                   attr_preds,
                   centernesses,
                   img_metas,
                   cfg=None,
                   rescale=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * 2. (bin = 2)
            depth_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * self.num_depth_cls.
            weights (list[Tensor]): Location-aware weights for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * self.weight_dim.
            attr_preds (list[Tensor]): Attribute scores for each scale level
                Has shape (N, num_points * num_attrs, H, W)
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config, optional): Test / postprocessing configuration,
                if None, test_cfg would be used. Defaults to None.
            rescale (bool, optional): If True, return boxes in original image
                space. Defaults to None.

        Returns:
            list[tuple[Tensor]]: Each item in result_list is a tuple, which
                consists of predicted 3D boxes, scores, labels, attributes and
                2D boxes (if necessary).
        """
        assert len(cls_scores) == len(bbox_preds) == len(dir_cls_preds) == \
            len(depth_cls_preds) == len(weights) == len(centernesses) == \
            len(attr_preds), 'The length of cls_scores, bbox_preds, ' \
            'dir_cls_preds, depth_cls_preds, weights, centernesses, and' \
            f'attr_preds: {len(cls_scores)}, {len(bbox_preds)}, ' \
            f'{len(dir_cls_preds)}, {len(depth_cls_preds)}, {len(weights)}' \
            f'{len(centernesses)}, {len(attr_preds)} are inconsistent.'
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            if self.use_direction_classifier:
                dir_cls_pred_list = [
                    dir_cls_preds[i][img_id].detach()
                    for i in range(num_levels)
                ]
            else:
                dir_cls_pred_list = [
                    cls_scores[i][img_id].new_full(
                        [2, *cls_scores[i][img_id].shape[1:]], 0).detach()
                    for i in range(num_levels)
                ]
            if self.use_depth_classifier:
                depth_cls_pred_list = [
                    depth_cls_preds[i][img_id].detach()
                    for i in range(num_levels)
                ]
            else:
                depth_cls_pred_list = [
                    cls_scores[i][img_id].new_full(
                        [self.num_depth_cls, *cls_scores[i][img_id].shape[1:]],
                        0).detach() for i in range(num_levels)
                ]
            if self.weight_dim != -1:
                weight_list = [
                    weights[i][img_id].detach() for i in range(num_levels)
                ]
            else:
                weight_list = [
                    cls_scores[i][img_id].new_full(
                        [1, *cls_scores[i][img_id].shape[1:]], 0).detach()
                    for i in range(num_levels)
                ]
            if self.pred_attrs:
                attr_pred_list = [
                    attr_preds[i][img_id].detach() for i in range(num_levels)
                ]
            else:
                attr_pred_list = [
                    cls_scores[i][img_id].new_full(
                        [self.num_attrs, *cls_scores[i][img_id].shape[1:]],
                        self.attr_background_label).detach()
                    for i in range(num_levels)
                ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            input_meta = img_metas[img_id]
            det_bboxes = self._get_bboxes_single(
                cls_score_list, bbox_pred_list, dir_cls_pred_list,
                depth_cls_pred_list, weight_list, attr_pred_list,
                centerness_pred_list, mlvl_points, input_meta, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           dir_cls_preds,
                           depth_cls_preds,
                           weights,
                           attr_preds,
                           centernesses,
                           mlvl_points,
                           input_meta,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * bbox_code_size, H, W).
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on a single scale level with shape
                (num_points * 2, H, W)
            depth_cls_preds (list[Tensor]): Box scores for probabilistic depth
                predictions on a single scale level with shape
                (num_points * self.num_depth_cls, H, W)
            weights (list[Tensor]): Location-aware weight maps on a single
                scale level with shape (num_points * self.weight_dim, H, W).
            attr_preds (list[Tensor]): Attribute scores for each scale level
                Has shape (N, num_points * num_attrs, H, W)
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 2).
            input_meta (dict): Metadata of input image.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool, optional): If True, return boxes in original image
                space. Defaults to False.

        Returns:
            tuples[Tensor]: Predicted 3D boxes, scores, labels, attributes and
                2D boxes (if necessary).
        """
        view = np.array(input_meta['cam2img'])
        scale_factor = input_meta['scale_factor']
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_centers2d = []
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        mlvl_attr_scores = []
        mlvl_centerness = []
        mlvl_depth_cls_scores = []
        mlvl_depth_uncertainty = []
        mlvl_bboxes2d = None
        if self.pred_bbox2d:
            mlvl_bboxes2d = []

        for cls_score, bbox_pred, dir_cls_pred, depth_cls_pred, weight, \
                attr_pred, centerness, points in zip(
                    cls_scores, bbox_preds, dir_cls_preds, depth_cls_preds,
                    weights, attr_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]
            depth_cls_pred = depth_cls_pred.permute(1, 2, 0).reshape(
                -1, self.num_depth_cls)
            depth_cls_score = F.softmax(
                depth_cls_pred, dim=-1).topk(
                    k=2, dim=-1)[0].mean(dim=-1)
            if self.weight_dim != -1:
                weight = weight.permute(1, 2, 0).reshape(-1, self.weight_dim)
            else:
                weight = weight.permute(1, 2, 0).reshape(-1, 1)
            depth_uncertainty = torch.exp(-weight[:, -1])
            attr_pred = attr_pred.permute(1, 2, 0).reshape(-1, self.num_attrs)
            attr_score = torch.max(attr_pred, dim=-1)[1]
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1,
                                                     sum(self.group_reg_dims))
            bbox_pred3d = bbox_pred[:, :self.bbox_coder.bbox_code_size]
            if self.pred_bbox2d:
                bbox_pred2d = bbox_pred[:, -4:]
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                merged_scores = scores * centerness[:, None]
                if self.use_depth_classifier:
                    merged_scores *= depth_cls_score[:, None]
                    if self.weight_dim != -1:
                        merged_scores *= depth_uncertainty[:, None]
                max_scores, _ = merged_scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred3d = bbox_pred3d[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_pred = dir_cls_pred[topk_inds, :]
                depth_cls_pred = depth_cls_pred[topk_inds, :]
                centerness = centerness[topk_inds]
                dir_cls_score = dir_cls_score[topk_inds]
                depth_cls_score = depth_cls_score[topk_inds]
                depth_uncertainty = depth_uncertainty[topk_inds]
                attr_score = attr_score[topk_inds]
                if self.pred_bbox2d:
                    bbox_pred2d = bbox_pred2d[topk_inds, :]
            # change the offset to actual center predictions
            bbox_pred3d[:, :2] = points - bbox_pred3d[:, :2]
            if rescale:
                bbox_pred3d[:, :2] /= bbox_pred3d[:, :2].new_tensor(
                    scale_factor)
                if self.pred_bbox2d:
                    bbox_pred2d /= bbox_pred2d.new_tensor(scale_factor)
            if self.use_depth_classifier:
                prob_depth_pred = self.bbox_coder.decode_prob_depth(
                    depth_cls_pred, self.depth_range, self.depth_unit,
                    self.division, self.num_depth_cls)
                sig_alpha = torch.sigmoid(self.fuse_lambda)
                bbox_pred3d[:, 2] = sig_alpha * bbox_pred3d[:, 2] + \
                    (1 - sig_alpha) * prob_depth_pred
            pred_center2d = bbox_pred3d[:, :3].clone()
            bbox_pred3d[:, :3] = points_img2cam(bbox_pred3d[:, :3], view)
            mlvl_centers2d.append(pred_center2d)
            mlvl_bboxes.append(bbox_pred3d)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)
            mlvl_depth_cls_scores.append(depth_cls_score)
            mlvl_attr_scores.append(attr_score)
            mlvl_centerness.append(centerness)
            mlvl_depth_uncertainty.append(depth_uncertainty)
            if self.pred_bbox2d:
                bbox_pred2d = distance2bbox(
                    points, bbox_pred2d, max_shape=input_meta['img_shape'])
                mlvl_bboxes2d.append(bbox_pred2d)

        mlvl_centers2d = torch.cat(mlvl_centers2d)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)
        if self.pred_bbox2d:
            mlvl_bboxes2d = torch.cat(mlvl_bboxes2d)

        # change local yaw to global yaw for 3D nms
        cam2img = torch.eye(
            4, dtype=mlvl_centers2d.dtype, device=mlvl_centers2d.device)
        cam2img[:view.shape[0], :view.shape[1]] = \
            mlvl_centers2d.new_tensor(view)
        mlvl_bboxes = self.bbox_coder.decode_yaw(mlvl_bboxes, mlvl_centers2d,
                                                 mlvl_dir_scores,
                                                 self.dir_offset, cam2img)

        mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            mlvl_bboxes,
            box_dim=self.bbox_coder.bbox_code_size,
            origin=(0.5, 0.5, 0.5)).bev)

        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_attr_scores = torch.cat(mlvl_attr_scores)
        mlvl_centerness = torch.cat(mlvl_centerness)
        # no scale_factors in box3d_multiclass_nms
        # Then we multiply it from outside
        mlvl_nms_scores = mlvl_scores * mlvl_centerness[:, None]
        if self.use_depth_classifier:  # multiply the depth confidence
            mlvl_depth_cls_scores = torch.cat(mlvl_depth_cls_scores)
            mlvl_nms_scores *= mlvl_depth_cls_scores[:, None]
            if self.weight_dim != -1:
                mlvl_depth_uncertainty = torch.cat(mlvl_depth_uncertainty)
                mlvl_nms_scores *= mlvl_depth_uncertainty[:, None]
        results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                       mlvl_nms_scores, cfg.score_thr,
                                       cfg.max_per_img, cfg, mlvl_dir_scores,
                                       mlvl_attr_scores, mlvl_bboxes2d)
        bboxes, scores, labels, dir_scores, attrs = results[0:5]
        attrs = attrs.to(labels.dtype)  # change data type to int
        bboxes = input_meta['box_type_3d'](
            bboxes,
            box_dim=self.bbox_coder.bbox_code_size,
            origin=(0.5, 0.5, 0.5))
        # Note that the predictions use origin (0.5, 0.5, 0.5)
        # Due to the ground truth centers2d are the gravity center of objects
        # v0.10.0 fix inplace operation to the input tensor of cam_box3d
        # So here we also need to add origin=(0.5, 0.5, 0.5)
        if not self.pred_attrs:
            attrs = None

        outputs = (bboxes, scores, labels, attrs)
        if self.pred_bbox2d:
            bboxes2d = results[-1]
            bboxes2d = torch.cat([bboxes2d, scores[:, None]], dim=1)
            outputs = outputs + (bboxes2d, )

        return outputs

    def get_targets(self, points, gt_bboxes_list, gt_labels_list,
                    gt_bboxes_3d_list, gt_labels_3d_list, centers2d_list,
                    depths_list, attr_labels_list):
        """Compute regression, classification and centerss targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
            gt_bboxes_3d_list (list[Tensor]): 3D Ground truth bboxes of each
                image, each has shape (num_gt, bbox_code_size).
            gt_labels_3d_list (list[Tensor]): 3D Ground truth labels of each
                box, each has shape (num_gt,).
            centers2d_list (list[Tensor]): Projected 3D centers onto 2D image,
                each has shape (num_gt, 2).
            depths_list (list[Tensor]): Depth of projected 3D centers onto 2D
                image, each has shape (num_gt, 1).
            attr_labels_list (list[Tensor]): Attribute labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        if attr_labels_list is None:
            attr_labels_list = [
                gt_labels.new_full(gt_labels.shape, self.attr_background_label)
                for gt_labels in gt_labels_list
            ]

        # get labels and bbox_targets of each image
        _, bbox_targets_list, labels_3d_list, bbox_targets_3d_list, \
            centerness_targets_list, attr_targets_list, gt_inds = multi_apply(
                self._get_target_single,
                gt_bboxes_list,
                gt_labels_list,
                gt_bboxes_3d_list,
                gt_labels_3d_list,
                centers2d_list,
                depths_list,
                attr_labels_list,
                points=concat_points,
                regress_ranges=concat_regress_ranges,
                num_points_per_lvl=num_points)

        label3d_test = labels_3d_list
        bbox_target3d_test = bbox_targets_3d_list
        centerness_targets_test = centerness_targets_list


        # split to per img, per level
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        labels_3d_list = [
            labels_3d.split(num_points, 0) for labels_3d in labels_3d_list
        ]
        bbox_targets_3d_list = [
            bbox_targets_3d.split(num_points, 0)
            for bbox_targets_3d in bbox_targets_3d_list
        ]
        centerness_targets_list = [
            centerness_targets.split(num_points, 0)
            for centerness_targets in centerness_targets_list
        ]
        attr_targets_list = [
            attr_targets.split(num_points, 0)
            for attr_targets in attr_targets_list
        ]
        # concat per level image
        concat_lvl_labels_3d = []
        concat_lvl_bbox_targets_3d = []
        concat_lvl_centerness_targets = []
        concat_lvl_attr_targets = []
        for i in range(num_levels):
            concat_lvl_labels_3d.append(
                torch.cat([labels[i] for labels in labels_3d_list]))
            concat_lvl_centerness_targets.append(
                torch.cat([
                    centerness_targets[i]
                    for centerness_targets in centerness_targets_list
                ]))
            bbox_targets_3d = torch.cat([
                bbox_targets_3d[i] for bbox_targets_3d in bbox_targets_3d_list
            ])
            if self.pred_bbox2d:
                bbox_targets = torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list])
                bbox_targets_3d = torch.cat([bbox_targets_3d, bbox_targets],
                                            dim=1)
            concat_lvl_attr_targets.append(
                torch.cat(
                    [attr_targets[i] for attr_targets in attr_targets_list]))
            if self.norm_on_bbox:
                bbox_targets_3d[:, :2] = \
                    bbox_targets_3d[:, :2] / self.strides[i]
                if self.pred_bbox2d:
                    bbox_targets_3d[:, -4:] = \
                        bbox_targets_3d[:, -4:] / self.strides[i]
            concat_lvl_bbox_targets_3d.append(bbox_targets_3d)

        return concat_lvl_labels_3d, concat_lvl_bbox_targets_3d, \
            concat_lvl_centerness_targets, concat_lvl_attr_targets, \
               label3d_test, bbox_target3d_test, centerness_targets_test, gt_inds

    def _get_target_single(self, gt_bboxes, gt_labels, gt_bboxes_3d,
                           gt_labels_3d, centers2d, depths, attr_labels,
                           points, regress_ranges, num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if not isinstance(gt_bboxes_3d, torch.Tensor):
            gt_bboxes_3d = gt_bboxes_3d.tensor.to(gt_bboxes.device)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.background_label), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_labels_3d.new_full(
                       (num_points,), self.background_label), \
                   gt_bboxes_3d.new_zeros((num_points, self.bbox_code_size)), \
                   gt_bboxes_3d.new_zeros((num_points,)), \
                   attr_labels.new_full(
                       (num_points,), self.attr_background_label)

        # change orientation to local yaw
        # todo 改变旋转角度
        gt_bboxes_3d[..., 6] = -torch.atan2(
            gt_bboxes_3d[..., 0], gt_bboxes_3d[..., 2]) + gt_bboxes_3d[..., 6]
        # todo 计算面积
        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # todo 扩充到点数
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        centers2d = centers2d[None].expand(num_points, num_gts, 2)
        gt_bboxes_3d = gt_bboxes_3d[None].expand(num_points, num_gts,
                                                 self.bbox_code_size)
        depths = depths[None, :, None].expand(num_points, num_gts, 1)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        delta_xs = (xs - centers2d[..., 0])[..., None]
        delta_ys = (ys - centers2d[..., 1])[..., None]
        # todo 构建3D目标
        bbox_targets_3d = torch.cat(
            (delta_xs, delta_ys, depths, gt_bboxes_3d[..., 3:]), dim=-1)
        # todo 计算2D每个点到真值的距离
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        assert self.center_sampling is True, 'Setting center_sampling to '\
            'False has not been implemented for FCOS3D.'
        # todo 当一个东西位于的真值的中心
        # condition1: inside a `center bbox`
        radius = self.center_sample_radius
        center_xs = centers2d[..., 0]
        center_ys = centers2d[..., 1]
        center_gts = torch.zeros_like(gt_bboxes)
        stride = center_xs.new_zeros(center_xs.shape)

        # project the points on current lvl back to the `original` sizes
        lvl_begin = 0
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
            lvl_begin = lvl_end

        center_gts[..., 0] = center_xs - stride
        center_gts[..., 1] = center_ys - stride
        center_gts[..., 2] = center_xs + stride
        center_gts[..., 3] = center_ys + stride

        cb_dist_left = xs - center_gts[..., 0]
        cb_dist_right = center_gts[..., 2] - xs
        cb_dist_top = ys - center_gts[..., 1]
        cb_dist_bottom = center_gts[..., 3] - ys
        center_bbox = torch.stack(
            (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
        # todo min(-1)[0] 表示返回最后的一个维度的最小值的具体值，
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        # todo 计算回归目标的在最后一个维度(到四个边界值的距离的)最大值三维具体值
        max_regress_distance = bbox_targets.max(-1)[0]
        # todo 限制回归的范围，最大的回归的范围的应该现在在每个层级中间
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))
        # center-based criterion to deal with ambiguity
        # todo 计算的3d目标的偏移量的距离
        dists = torch.sqrt(torch.sum(bbox_targets_3d[..., :2]**2, dim=-1))
        # todo 将不位于真值里面的距离设置为无限大
        dists[inside_gt_bbox_mask == 0] = INF
        # todo 将不位于的回归的范围里面的设置为无限大
        dists[inside_regress_range == 0] = INF
        # todo 计算每一个样本点到所有的真值的距离的，第一个返回的是距离， 第二个返回的是最大值的索引
        min_dist, min_dist_inds = dists.min(dim=1)

        # todo 为每个样本赋予对应的标签
        labels = gt_labels[min_dist_inds]
        labels_3d = gt_labels_3d[min_dist_inds]
        attr_labels = attr_labels[min_dist_inds]

        # todo 为所有的负样本赋予背景的标签
        labels[min_dist == INF] = self.background_label  # set as BG
        labels_3d[min_dist == INF] = self.background_label  # set as BG
        attr_labels[min_dist == INF] = self.attr_background_label

        # todo 为每一个物体都生成一个标志位，这个代表的是他对应的真值
        gt_inds = min_dist_inds.clone().detach()
        gt_inds[labels_3d == self.background_label] = -1

        bbox_targets = bbox_targets[range(num_points), min_dist_inds]
        bbox_targets_3d = bbox_targets_3d[range(num_points), min_dist_inds]
        relative_dists = torch.sqrt(
            torch.sum(bbox_targets_3d[..., :2]**2,
                      dim=-1)) / (1.414 * stride[:, 0])

        # todo 生成中心度的目标 [39780]
        # [N, 1] / [N, 1]
        centerness_targets = torch.exp(-self.centerness_alpha * relative_dists)
        # todo 计算 3diou axis_aligned_bbox_overlaps_3d
        # giou_targets = axis_aligned_bbox_overlaps_3d([], [])

        return labels, bbox_targets, labels_3d, bbox_targets_3d, \
            centerness_targets, attr_labels, gt_inds

