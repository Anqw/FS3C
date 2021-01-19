# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torch
from torch import nn
import numpy as np
import math

from fs3c.structures import ImageList
from fs3c.utils.logger import log_first_n

from ..backbone import build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads, build_jig_heads, build_rot_heads
from .build import META_ARCH_REGISTRY
import torchvision.transforms as transforms

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.jig_heads = build_jig_heads(cfg, self.backbone.output_shape())
        self.rot_heads = build_rot_heads(cfg, self.backbone.output_shape())
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES

        self.jigsaw = cfg.JIG
        self.rotation = cfg.ROT
        self.ssl = cfg.SSL
        self.tensor_to_img = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.ToPILImage()])
        transform_list = ['RandomResizedCrop', 'RandomHorizontalFlip', 'ToTensor']
        transform_funcs = [self.parse_transform(x) for x in transform_list]
        self.transform = transforms.Compose(transform_funcs)
        self.transform_jigsaw = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(255,scale=(0.5, 1.0)),
                    transforms.RandomHorizontalFlip()])
        self.transform_patch_jigsaw = transforms.Compose([
                transforms.RandomCrop(64),
                transforms.Lambda(self.rgb_jittering),
                transforms.ToTensor()])
        self.permutations = np.load('permutations_35.npy')
        if self.permutations.min() == 1:
            self.permutations = self.permutations - 1
        self.pool_jig = nn.AdaptiveMaxPool2d((1, 1))

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print('froze backbone parameters')

        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print('froze proposal generator parameters')

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = False
            print('froze roi_box_head parameters')

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """
        if not self.training:
            return self.inference(batched_inputs[0])

        images = self.preprocess_image(batched_inputs[0])
        if "instances" in batched_inputs[0][0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs[0]]
        elif "targets" in batched_inputs[0][0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs[0]]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        ssl_rot_losses = 0
        ssl_jig_losses = 0
        if self.jigsaw:
            patches, labels = self.get_patches(batched_inputs[1], self.transform_jigsaw, self.transform_patch_jigsaw,
                                              self.permutations)
            B, T, C, H, W = patches.size()
            patches = patches.view(B*T, C, H, W)
            features_ssl = self.backbone(patches)
            features_ssl = features_ssl["p6"]
            features_ssl = torch.flatten(features_ssl, start_dim=1).view(B, T, -1)
            features_ssl = features_ssl.transpose(0, 1)
            image = None
            proposal = None
            ssl_jig_losses = self.jig_heads(image, features_ssl, proposal, labels)

        if self.rotation:
            images_ = [x["image"].to(self.device) for x in batched_inputs[1]]
            data_list = []
            order_list = []
            for img in images_:
                img = img.cpu()
                img = img.permute(1, 2, 0).numpy()
                img = self.tensor_to_img(img)
                rotated_imgs = [
                    self.transform(img),
                    self.transform(img.rotate(90, expand=True)),
                    self.transform(img.rotate(180, expand=True)),
                    self.transform(img.rotate(270, expand=True))
                ]
                rotated_img = [x*255 for x in rotated_imgs]
                rotated_img = [self.normalizer(x.to(self.device)) for x in rotated_img]
                rotation_labels = torch.Tensor([0, 1, 2, 3])
                data = torch.stack(rotated_img, dim=0)
                data_list.append(data)
                order_list.append(rotation_labels)
            patches = torch.stack(data_list, 0).to(self.device)
            labels = torch.cat(order_list).long().to(self.device)

            B, R, C, H, W = patches.size()
            patches = patches.view(B * R, C, H, W)
            features_ssl = self.backbone(patches)
            features_ssl = features_ssl["p6"]
            image = None
            proposal = None
            ssl_rot_losses = self.rot_heads(image, features_ssl, proposal, labels)
        if self.ssl:
            ssl_losses = {"ssl_rot_losses": 0.5 * ssl_rot_losses,
                          "ssl_jig_losses": 0.5 * ssl_jig_losses,
                          }

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0][0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs[0]]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        losses = {}
        losses.update(detector_losses)
        if self.ssl:
            losses.update(ssl_losses)
        losses.update(proposal_losses)
        return losses

    def rgb_jittering(self, im):
        im = np.array(im, 'int32')
        for ch in range(3):
            im[:, :, ch] += np.random.randint(-2, 2)
        im[im > 255] = 255
        im[im < 0] = 0
        return im.astype('uint8')

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def get_patches(self, batched_inputs, transform_jigsaw, transform_patch_jigsaw, permutations):
        images = [x["image"].to(self.device) for x in batched_inputs]
        data_list=[]
        order_list=[]
        for img in images:
            img = img.cpu()
            img = img.permute(1, 2, 0).numpy()
            img = transform_jigsaw(img)
            s = float(img.size[0]) / 3
            a = s / 2
            tiles = [None] * 9
            for n in range(9):
                i = int(n / 3)
                j = n % 3
                c = [a * i * 2 + a, a * j * 2 + a]
                c = np.array([math.ceil(c[1] - a), math.ceil(c[0] - a), int(c[1] + a), int(c[0] + a)]).astype(int)
                tile = img.crop(c.tolist())
                tile = transform_patch_jigsaw(tile)
                tile = tile * 255
                # Normalize the patches indipendently to avoid low level features shortcut
                m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
                s[s == 0] = 1
                norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
                tile = norm(tile)
                tiles[n] = tile

            order = np.random.randint(len(permutations))
            data = [tiles[permutations[order][t]] for t in range(9)]
            data = torch.stack(data, 0)
            data_list.append(data)
            order_list.append(int(order))
        data_ssl = torch.stack(data_list, 0).to(self.device)
        order_ssl = torch.Tensor(order_list).long().to(self.device)

        return data_ssl, order_ssl

    def parse_transform(self, transform_type):
        method = getattr(transforms, transform_type)
        if transform_type == 'RandomResizedCrop':
            return method(224)
        elif transform_type == 'CenterCrop':
            return method(224)
        elif transform_type == 'Resize':
            return method(224)
        else:
            return method()


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
