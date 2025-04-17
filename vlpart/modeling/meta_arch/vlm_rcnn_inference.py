# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.config import configurable
from detectron2.layers import move_device_like, batched_nms
from detectron2.structures import ImageList, ROIMasks, Instances, Boxes

from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from torch.cuda.amp import autocast

from ..text_encoder.text_encoder import build_text_encoder
from ..utils.detic import load_class_freq


@META_ARCH_REGISTRY.register()
class VLMRCNNInference(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        eval_proposal=False,
        with_image_labels=False,
        fp16=False,
        sync_caption_batch=False,
        roi_head_name='',
        cap_batch_ratio=4,
        with_caption=False,
        text_encoder_type="ViT-B/32",
        text_encoder_dim=512,
        dynamic_classifier=False,
        **kwargs
    ):
        super().__init__()

        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.eval_proposal = eval_proposal
        self.with_image_labels = with_image_labels
        self.fp16 = fp16
        self.with_caption = with_caption
        self.sync_caption_batch = sync_caption_batch
        self.roi_head_name = roi_head_name
        self.cap_batch_ratio = cap_batch_ratio
        self.text_encoder_dim = text_encoder_dim

        self.dynamic_classifier = dynamic_classifier
        if self.dynamic_classifier:
            self.freq_weight = kwargs.pop('freq_weight')
            self.num_classes = kwargs.pop('num_classes')
            self.num_sample_cats = kwargs.pop('num_sample_cats')

        if self.with_caption:
            assert not self.dynamic_classifier
            self.text_encoder = build_text_encoder(
                pretrain=True, visual_type=text_encoder_type)
            for v in self.text_encoder.parameters():
                v.requires_grad = False

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
                self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        ret = {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

        ret.update({
            'eval_proposal': cfg.MODEL.EVAL_PROPOSAL,
            'with_image_labels': cfg.WITH_IMAGE_LABELS,
            'fp16': cfg.FP16,
            'with_caption': cfg.MODEL.WITH_CAPTION,
            'sync_caption_batch': cfg.MODEL.SYNC_CAPTION_BATCH,
            'dynamic_classifier': cfg.MODEL.DYNAMIC_CLASSIFIER,
            'roi_head_name': cfg.MODEL.ROI_HEADS.NAME,
            'cap_batch_ratio': cfg.MODEL.CAP_BATCH_RATIO,
            'text_encoder_type': cfg.MODEL.TEXT_ENCODER_TYPE,
            'text_encoder_dim': cfg.MODEL.TEXT_ENCODER_DIM,
        })

        if ret['dynamic_classifier']:
            ret['freq_weight'] = load_class_freq(
                cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH,
                cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT)
            ret['num_classes'] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
            ret['num_sample_cats'] = cfg.MODEL.NUM_SAMPLE_CATS

        return ret

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], proposals: list[Instances]):
        if not self.training:
            return self.inference(batched_inputs, proposals)
        else:
            raise NotImplementedError("Training is not implemented")

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], proposals: list[Instances]):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        results = self.roi_heads(images, features, proposals, return_scores_only=True)

        return results

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        original_images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in original_images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images
