# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Detectron2 - A next-generation platform for object detection and segmentation.

This package provides:
- Pre-trained models for object detection, instance segmentation, semantic
  segmentation, panoptic segmentation, and keypoint detection.
- A flexible framework for building custom detection and segmentation models.
- Efficient training and inference pipelines.

Example usage::

    import detectron2
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg

    cfg = get_cfg()
    cfg.merge_from_file("path/to/config.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(image)

Notes:
    Personal fork for experimenting with custom dataset training and
    fine-tuning pre-trained models on domain-specific data.
"""

from .version import __version__

__all__ = ["__version__"]
