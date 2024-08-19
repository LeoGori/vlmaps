import sys
import os

# Add the directory containing the gsam folder to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

import supervision as sv
import torch

# segment anything
from segment_anything.segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add the directory containing the gsam folder to the Python path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'vlmaps'))

from huggingface_hub import hf_hub_download
import os
from PIL import Image
from vlmaps.lseg.additional_utils.models import resize_image, pad_image, crop_image
import math
from vlmaps.utils.mapping_utils import *
from torchvision.ops import box_convert


class GSAM:

    def __init__(self, device):
        print("GSAM initialized")
        self.transform = None
        # check that self.transform is None
        self.device = device

        self.groundingdino_model = GDINO(device)
        self.sam_predictor = SAM(device)

        self._init_gsam()

    
    def _init_gsam(self):
        self.crop_size = 512  # 480
        self.base_size = 512  # 520

        # todo: most porbably grounding dino does not normalization of the image, need to check
        self.norm_mean = [0.5, 0.5, 0.5]
        self.norm_std = [0.5, 0.5, 0.5]
        
        transform = self.get_transform()

        return self.sam_predictor, transform, self.crop_size, self.base_size, self.norm_mean, self.norm_std
        # return self.sam_predictor, transform, self.crop_size, self.base_size
    
    def get_transform(self):
        if self.transform is None:
            self.transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        return self.transform

    def load_image(self, image_path: str) -> tuple[np.array, torch.Tensor]:
        image_source = Image.open(image_path).convert("RGB")

        image, image_transformed = self.preprocess_image(image_source)
        
        return image, image_transformed
    
    def preprocess_image(self, image_source) -> tuple[np.array, torch.Tensor]:
        transform = self.get_transform()
        image = np.asarray(image_source)
        image_transformed, _ = transform(image_source, None)
        return image, image_transformed
    
    def get_sam_feat(self,
        image: np.array,
        labels,
        boxes,
        # norm_mean=[0.5, 0.5, 0.5],
        # norm_std=[0.5, 0.5, 0.5],
        vis=False,
    ):
        # vis_image = image.copy()
        # image = self.transform(image).unsqueeze(0).to(self.device)
        # img = image[0].permute(1, 2, 0)
        # img = img * 0.5 + 0.5

        # image, _ = self.transform(image, None)
        # add batch dim
        image = image.unsqueeze(0)

        batch, _, h, w = image.size()
        stride_rate = 2.0 / 3.0
        stride = int(self.crop_size * stride_rate)

        # long_size = int(math.ceil(base_size * scale))
        long_size = self.base_size
        if h > w:
            height = long_size
            width = int(1.0 * w * long_size / h + 0.5)
            short_size = width
        else:
            width = long_size
            height = int(1.0 * h * long_size / w + 0.5)
            short_size = height

        cur_img = resize_image(image, height, width, **{"mode": "bilinear", "align_corners": True})

        if long_size <= self.crop_size:
            pad_img = pad_image(cur_img, self.norm_mean, self.norm_std, self.crop_size)
            print(pad_img.shape)
            with torch.no_grad():
                # outputs = model(pad_img)
                # remove batch dim
                pad_img = pad_img.squeeze(0)
                outputs, logits = self.sam_predictor.predict_masks_batch(pad_img, boxes)
            
            # get the indices of the list outputs that are not None
            indices = [i for i, x in enumerate(outputs) if x is not None]
            outputs = [outputs[i] for i in indices]
            logits = [logits[i] for i in indices]
            labels = [labels[i] for i in indices]

            # outputs are segmentation boolean masks, logits are the quality of the masks and labels are the labels of the masks
            # overwrite outputs with the values of the corresponding labels where outputs is true
            for i, output in enumerate(outputs):
                for pixel_mask in output:
                    print(pixel_mask.shape)
                    if pixel_mask:
                        output[pixel_mask] = labels[i]
                            
            # merge the outputs in a single image with the same size as the original image
            outputs = torch.stack(outputs)


            outputs = crop_image(outputs, 0, height, 0, width)
        else:
            if short_size < self.crop_size:
                # pad if needed
                pad_img = pad_image(cur_img, self.norm_mean, self.norm_std, self.crop_size)
            else:
                pad_img = cur_img
            _, _, ph, pw = pad_img.shape  # .size()
            assert ph >= height and pw >= width
            h_grids = int(math.ceil(1.0 * (ph - self.crop_size) / stride)) + 1
            w_grids = int(math.ceil(1.0 * (pw - self.crop_size) / stride)) + 1
            with torch.cuda.device_of(image):
                with torch.no_grad():
                    # outputs = image.new().resize_(batch, self.sam_predictor.out_c, ph, pw).zero_().to(self.device)
                    outputs = image.new().resize_(batch, labels.shape[2], ph, pw).zero_().to(self.device)
                    if vis:
                        logits_outputs = image.new().resize_(batch, len(labels), ph, pw).zero_().to(self.device)
                count_norm = image.new().resize_(batch, 1, ph, pw).zero_().to(self.device)
            # grid evaluation
            for idh in range(h_grids):
                for idw in range(w_grids):
                    h0 = idh * stride
                    w0 = idw * stride
                    h1 = min(h0 + self.crop_size, ph)
                    w1 = min(w0 + self.crop_size, pw)
                    crop_img = crop_image(pad_img, h0, h1, w0, w1)
                    # pad if needed
                    pad_crop_img = pad_image(crop_img, self.norm_mean, self.norm_std, self.crop_size)
                    with torch.no_grad():
                        # output = model(pad_crop_img)
                        output, logits = self.sam_predictor_masks_batch(pad_crop_img, boxes)
                    cropped = crop_image(output, 0, h1 - h0, 0, w1 - w0)
                    outputs[:, :, h0:h1, w0:w1] += cropped
                    if vis:
                        cropped_logits = crop_image(logits, 0, h1 - h0, 0, w1 - w0)
                        logits_outputs[:, :, h0:h1, w0:w1] += cropped_logits
                    count_norm[:, :, h0:h1, w0:w1] += 1
            assert (count_norm == 0).sum() == 0
            outputs = outputs / count_norm
            outputs = outputs[:, :, :height, :width]
            if vis:
                logits_outputs = logits_outputs / count_norm
                logits_outputs = logits_outputs[:, :, :height, :width]
        # outputs = resize_image(outputs, h, w, **{'mode': 'bilinear', 'align_corners': True})
        # outputs = resize_image(outputs, image.shape[0], image.shape[1], **{'mode': 'bilinear', 'align_corners': True})
        outputs = outputs.cpu()
        outputs = outputs.numpy()  # B, D, H, W
        if vis:
            predicts = [torch.max(logit, 0)[1].cpu().numpy() for logit in logits_outputs]
            pred = predicts[0]
            new_palette = get_new_pallete(len(labels))
            mask, patches = get_new_mask_pallete(pred, new_palette, out_label_flag=True, labels=labels)
            seg = mask.convert("RGBA")
            cv2.imshow("image", vis_image[:, :, [2, 1, 0]])
            #cv2.waitKey()
            fig = plt.figure()
            plt.imshow(seg)
            plt.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.0, 1), prop={"size": 20})
            plt.axis("off")

            plt.tight_layout()
            plt.show()
            cv2.waitKey()

        return outputs
    

class GDINO():

    def __init__(self, device) -> None:
        # load grounding dino

        # Use this command for evaluate the Grounding DINO model
        # Or you can download the model by yourself
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

        self.device = torch.device(device)

        def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
            cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

            args = SLConfig.fromfile(cache_config_file) 
            model = build_model(args)

            cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
            checkpoint = torch.load(cache_file, map_location='cpu')
            log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
            print("Model loaded from {} \n => {}".format(cache_file, log))
            _ = model.eval()
            return model
        
        self.model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, self.device)

    def predict_single_caption(
        self,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        caption = self.__preprocess_caption(caption=caption)

        self.model = self.model.to(self.device)
        image = image.to(self.device)

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(image[None], captions=[caption])

        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
        prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

        mask = prediction_logits.max(dim=1)[0] > box_threshold
        logits = prediction_logits[mask]  # logits.shape = (n, 256)
        boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

        tokenizer = self.model.tokenizer
        tokenized = tokenizer(caption)

        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit
            in logits
        ]

        return boxes, logits.max(dim=1)[0], phrases, tokenized
    
    def __preprocess_caption(self, caption: str) -> str:
        result = caption.lower().strip()
        if result.endswith("."):
            return result
        return result + "."

    def predict_captions(self, image, captions):
        boxes = []
        scores = []
        phrases = []
        tokenizeds = []

        for caption in captions:
            # check the shape of the box, if it is empty, then this means that the caption is not detected, so we can skip it

            box, score, phrase, tokenized = self.predict_single_caption(image, caption, 0.5, 0.5)

            boxes.append(box)
            scores.append(score)
            phrases.append(phrase)
            tokenizeds.append(tokenized)

        return boxes, scores, phrases, tokenizeds

    def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: list[str]) -> np.ndarray:
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        detections = sv.Detections(xyxy=xyxy)

        labels = [
            f"{phrase} {logit:.2f}"
            for phrase, logit
            in zip(phrases, logits)
        ]

        return detections, labels
    

class SAM:
    
    def __init__(self, device):
        self.device = device

        sam_checkpoint = 'sam_vit_h_4b8939.pth'
        sam = build_sam(checkpoint=sam_checkpoint)
        sam.to(device)
        self.model = SamPredictor(sam)

    def predict(self, image_source, boxes):
        
        # set image
        self.model.set_image(image_source)

        # todo: check this part with the normalization kept in vlmap_builder
        # box: normalized box xywh -> unnormalized xyxy
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        transformed_boxes = self.model.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(self.device)

        # Arguments:
        #   point_coords (np.ndarray or None): A Nx2 array of point prompts to the
        #     model. Each point is in (X,Y) in pixels.
        #   point_labels (np.ndarray or None): A length N array of labels for the
        #     point prompts. 1 indicates a foreground point and 0 indicates a
        #     background point.
        #   box (np.ndarray or None): A length 4 array given a box prompt to the
        #     model, in XYXY format.
        #   mask_input (np.ndarray): A low resolution mask input to the model, typically
        #     coming from a previous prediction iteration. Has form 1xHxW, where
        #     for SAM, H=W=256.
        #   multimask_output (bool): If true, the model will return three masks.
        #     For ambiguous input prompts (such as a single click), this will often
        #     produce better masks than a single prediction. If only a single
        #     mask is needed, the model's predicted quality score can be used
        #     to select the best mask. For non-ambiguous prompts, such as multiple
        #     input prompts, multimask_output=False can give better results.
        #   return_logits (bool): If true, returns un-thresholded masks logits
        #     instead of a binary mask.

        # Returns:
        #   (np.ndarray): The output masks in CxHxW format, where C is the
        #     number of masks, and (H, W) is the original image size.
        #   (np.ndarray): An array of length C containing the model's
        #     predictions for the quality of each mask.
        #   (np.ndarray): An array of shape CxHxW, where C is the number
        #     of masks and H=W=256. These low resolution logits can be passed to
        #     a subsequent iteration as mask input.
        masks, logits, _ = self.model.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        )

        return masks, logits
    
    # can be optimized not preprocessing the image everytime
    def predict_masks_batch(self, image, boxes):
        masks = []
        logits = []
        for box in boxes:
            if len(box) == 0:
                masks.append(None)
                logits.append(None)
            else:
                mask, logit = self.predict(image, box)
                masks.append(mask)
                logits.append(logit)
        return masks, logits