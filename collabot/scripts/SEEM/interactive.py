# --------------------------------------------------------
# SEEM -- Segment Everything Everywhere All At Once
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.visualizer import Visualizer
from detectron2.utils.colormap import random_color
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks
from modeling.language.loss import vl_similarity
from utils.constants import COCO_PANOPTIC_CLASSES
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

import cv2
import os
import glob
import subprocess
from PIL import Image
import random

t = []
t.append(transforms.Resize(720, interpolation=Image.BICUBIC))
transform = transforms.Compose(t)
metadata = MetadataCatalog.get('coco_2017_train_panoptic')
all_classes = [name.replace('-other','').replace('-merged','') for name in COCO_PANOPTIC_CLASSES] + ["others"]
colors_list = [(np.array(color['color'])/255).tolist() for color in COCO_CATEGORIES] + [[1, 1, 1]]

def interactive_infer_image(model, audio_model, image, tasks, refimg=None, reftxt=None, audio_pth=None, video_pth=None):
    image_ori = transform(image['image'])
    mask_ori = image['mask']
    width = image_ori.size[0]
    height = image_ori.size[1]
    image_ori = np.asarray(image_ori)
    visual = Visualizer(image_ori, metadata=metadata)
    images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()

    # stroke_inimg = None
    # stroke_refimg = None

    data = {"image": images, "height": height, "width": width}
    if len(tasks) == 0:
        tasks = ["Panoptic"]
    
    # inistalize task
    model.model.task_switch['spatial'] = False
    model.model.task_switch['visual'] = False
    model.model.task_switch['grounding'] = False
    model.model.task_switch['audio'] = False

    example = None
    if 'Example' in tasks:
        model.model.task_switch['visual'] = True
        model.model.task_switch['spatial'] = True
        refimg_ori, refimg_mask = refimg['image'], refimg['mask']
        refimg_ori = transform(refimg_ori)
        _width = refimg_ori.size[0]
        _height = refimg_ori.size[1]
        refimg_ori = np.asarray(refimg_ori)
        refimg_ori_np = refimg_ori.copy()
        images = torch.from_numpy(refimg_ori.copy()).permute(2,0,1).cuda()
        batched_inputs = [{'image': images, 'height': _height, 'width': _width, 'spatial_query':{}}]

        refimg_mask = np.asarray(refimg_mask)[:,:,0:1].copy()
        refimg_mask = torch.from_numpy(refimg_mask).permute(2,0,1)[None,]
        refimg_mask = (F.interpolate(refimg_mask, (_height, _width), mode='bilinear') > 0)
        batched_inputs[0]['spatial_query']['rand_shape'] = refimg_mask
        outputs_refimg, img_shape = model.model.evaluate_referring_image(batched_inputs)
        model.model.task_switch['spatial'] = False
        data['visual'] = outputs_refimg

        # overlay = refimg_mask[0,0].float().numpy()[:,:,None] * np.array([0,0,255])
        # x = refimg_ori_np
        # stroke_refimg = x * (1 - refimg_mask[0,0].float().numpy()[:,:,None]) + (x * refimg_mask[0,0].numpy()[:,:,None] * 0.2 + overlay * 0.8)
        # stroke_refimg = Image.fromarray(stroke_refimg.astype(np.uint8))

    stroke = None
    if 'Stroke' in tasks:
        model.model.task_switch['spatial'] = True
        mask_ori = np.asarray(mask_ori)[:,:,0:1].copy()
        mask_ori = torch.from_numpy(mask_ori).permute(2,0,1)[None,]
        mask_ori = (F.interpolate(mask_ori, (height, width), mode='bilinear') > 0)
        data['stroke'] = mask_ori

        # overlay = mask_ori[0,0].float().numpy()[:,:,None] * np.array([0,255,0])
        # x = image_ori
        # stroke_inimg = x * (1 - mask_ori[0,0].float().numpy()[:,:,None]) + (x * mask_ori[0,0].numpy()[:,:,None] * 0.2 + overlay * 0.8)
        # stroke_inimg = Image.fromarray(stroke_inimg.astype(np.uint8))

    text = None
    if 'Text' in tasks:
        model.model.task_switch['grounding'] = True
        data['text'] = [reftxt]

    audio = None
    if 'Audio' in tasks:
        model.model.task_switch['audio'] = True
        audio_result = audio_model.transcribe(audio_pth)
        data['audio'] = [audio_result['text']]

    batch_inputs = [data]
    if 'Panoptic' in tasks:
        model.model.metadata = metadata
        results = model.model.evaluate(batch_inputs)
        pano_seg = results[-1]['panoptic_seg'][0]
        pano_seg_info = results[-1]['panoptic_seg'][1]
        demo = visual.draw_panoptic_seg(pano_seg.cpu(), pano_seg_info) # rgb Image
        res = demo.get_image()
        return Image.fromarray(res), None
    else:
        results,image_size,extra = model.model.evaluate_demo(batch_inputs)

    # If contians spatial use spatial:
    if 'Stroke' in tasks:
        v_emb = results['pred_maskembs']
        s_emb = results['pred_pspatials']
        pred_masks = results['pred_masks']

        pred_logits = v_emb @ s_emb.transpose(1,2)
        logits_idx_y = pred_logits[:,:,0].max(dim=1)[1]
        logits_idx_x = torch.arange(len(logits_idx_y), device=logits_idx_y.device)
        logits_idx = torch.stack([logits_idx_x, logits_idx_y]).tolist()
        pred_masks_pos = pred_masks[logits_idx]
        pred_class = results['pred_logits'][logits_idx].max(dim=-1)[1]

    elif 'Example' in tasks:
        v_emb = results['pred_maskembs']
        s_emb = results['pred_pvisuals']
        pred_masks = results['pred_masks']

        pred_logits = v_emb @ s_emb.transpose(1,2)
        logits_idx_y = pred_logits[:,:,0].max(dim=1)[1]
        logits_idx_x = torch.arange(len(logits_idx_y), device=logits_idx_y.device)
        logits_idx = torch.stack([logits_idx_x, logits_idx_y]).tolist()
        pred_masks_pos = pred_masks[logits_idx]
        pred_class = results['pred_logits'][logits_idx].max(dim=-1)[1]
    
    elif 'Text' in tasks:
        pred_masks = results['pred_masks'][0]
        v_emb = results['pred_captions'][0]
        t_emb = extra['grounding_class']

        t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)

        temperature = model.model.sem_seg_head.predictor.lang_encoder.logit_scale
        out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)
        
        matched_id = out_prob.max(0)[1]
        pred_masks_pos = pred_masks[matched_id,:,:]
        pred_class = results['pred_logits'][0][matched_id].max(dim=-1)[1]

    elif 'Audio' in tasks:
        pred_masks = results['pred_masks'][0]
        v_emb = results['pred_captions'][0]
        t_emb = extra['audio_class']

        t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)

        temperature = model.model.sem_seg_head.predictor.lang_encoder.logit_scale
        out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)
        
        matched_id = out_prob.max(0)[1]
        pred_masks_pos = pred_masks[matched_id,:,:]
        pred_class = results['pred_logits'][0][matched_id].max(dim=-1)[1]

    # interpolate mask to ori size
    pred_masks_pos = (F.interpolate(pred_masks_pos[None,], image_size[-2:], mode='bilinear')[0,:,:data['height'],:data['width']] > 0.0).float().cpu().numpy()
    texts = [all_classes[pred_class[0]]]

    for idx, mask in enumerate(pred_masks_pos):
        # color = random_color(rgb=True, maximum=1).astype(np.int32).tolist()
        out_txt = texts[idx] if 'Text' not in tasks else reftxt
        demo = visual.draw_binary_mask(mask, color=colors_list[pred_class[0]%133], text=out_txt)
    res = demo.get_image()
    torch.cuda.empty_cache()
    # return Image.fromarray(res), stroke_inimg, stroke_refimg
    return pred_masks_pos, Image.fromarray(res)
