import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)


import sys
import torch

sys.path.append(os.path.abspath('../catseg'))

import cat_seg



from detectron2.config import CfgNode as CN
import detectron2.utils.comm as comm
from contextlib import contextmanager

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


def add_cat_seg_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"

    cfg.DATASETS.VAL_ALL = ("coco_2017_val_all_stuff_sem_seg",)

    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]

    # zero shot config
    cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON = "datasets/ADE20K_2021_17_01/ADE20K_847.json"
    cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON = "datasets/ADE20K_2021_17_01/ADE20K_847.json"
    cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_INDEXES = "datasets/coco/coco_stuff/split/seen_indexes.json"
    cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_INDEXES = "datasets/coco/coco_stuff/split/unseen_indexes.json"

    cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED = "ViT-B/16"

    cfg.MODEL.PROMPT_ENSEMBLE = False
    cfg.MODEL.PROMPT_ENSEMBLE_TYPE = "single"

    cfg.MODEL.CLIP_PIXEL_MEAN = [122.7709383, 116.7460125, 104.09373615]
    cfg.MODEL.CLIP_PIXEL_STD = [68.5005327, 66.6321579, 70.3231630]
    # three styles for clip classification, crop, mask, cropmask

    cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM = 512
    cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM = 128
    cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_DIM = 512
    cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_PROJ_DIM = 128

    cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS = [64, 32]
    cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS = [256, 128]
    cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS = [32, 16]

    cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS = 4
    cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS = 4
    cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIMS = 128
    cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES = [6, 6]
    cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION = [24, 24]
    cfg.MODEL.SEM_SEG_HEAD.WINDOW_SIZES = 12
    cfg.MODEL.SEM_SEG_HEAD.ATTENTION_TYPE = "linear"

    cfg.MODEL.SEM_SEG_HEAD.PROMPT_DEPTH = 0
    cfg.MODEL.SEM_SEG_HEAD.PROMPT_LENGTH = 0
    cfg.SOLVER.CLIP_MULTIPLIER = 0.01

    cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE = "attention"
    cfg.TEST.SLIDING_WINDOW = False




def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask_former")
    return cfg


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to DETR.
    """

    # @classmethod
    # def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    #     """
    #     Create evaluator(s) for a given dataset.
    #     This uses the special metadata "evaluator_type" associated with each
    #     builtin dataset. For your own dataset, you can simply create an
    #     evaluator manually in your script and do not have to worry about the
    #     hacky if-else logic here.
    #     """
    #     if output_folder is None:
    #         output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    #     evaluator_list = []
    #     evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    #     if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
    #         evaluator_list.append(
    #             SemSegEvaluator(
    #                 dataset_name,
    #                 distributed=True,
    #                 output_dir=output_folder,
    #             )
    #         )

    #     if evaluator_type == "sem_seg_background":
    #         evaluator_list.append(
    #             VOCbEvaluator(
    #                 dataset_name,
    #                 distributed=True,
    #                 output_dir=output_folder,
    #             )
    #         )
    #     if evaluator_type == "coco":
    #         evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    #     if evaluator_type in [
    #         "coco_panoptic_seg",
    #         "ade20k_panoptic_seg",
    #         "cityscapes_panoptic_seg",
    #     ]:
    #         evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    #     if evaluator_type == "cityscapes_instance":
    #         assert (
    #             torch.cuda.device_count() >= comm.get_rank()
    #         ), "CityscapesEvaluator currently do not work with multiple machines."
    #         return CityscapesInstanceEvaluator(dataset_name)
    #     if evaluator_type == "cityscapes_sem_seg":
    #         assert (
    #             torch.cuda.device_count() >= comm.get_rank()
    #         ), "CityscapesEvaluator currently do not work with multiple machines."
    #         return CityscapesSemSegEvaluator(dataset_name)
    #     if evaluator_type == "cityscapes_panoptic_seg":
    #         assert (
    #             torch.cuda.device_count() >= comm.get_rank()
    #         ), "CityscapesEvaluator currently do not work with multiple machines."
    #         evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
    #     if len(evaluator_list) == 0:
    #         raise NotImplementedError(
    #             "no Evaluator for the dataset {} with the type {}".format(
    #                 dataset_name, evaluator_type
    #             )
    #         )
    #     elif len(evaluator_list) == 1:
    #         return evaluator_list[0]
    #     return DatasetEvaluators(evaluator_list)

    # @classmethod
    # def build_train_loader(cls, cfg):
    #     # Semantic segmentation dataset mapper
    #     if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
    #         mapper = MaskFormerSemanticDatasetMapper(cfg, True)
    #     # Panoptic segmentation dataset mapper
    #     elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
    #         mapper = MaskFormerPanopticDatasetMapper(cfg, True)
    #     # DETR-style dataset mapper for COCO panoptic segmentation
    #     elif cfg.INPUT.DATASET_MAPPER_NAME == "detr_panoptic":
    #         mapper = DETRPanopticDatasetMapper(cfg, True)
    #     else:
    #         mapper = None
    #     return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        # import ipdb;
        # ipdb.set_trace()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if "clip_model" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.CLIP_MULTIPLIER
                # for deformable detr

                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res