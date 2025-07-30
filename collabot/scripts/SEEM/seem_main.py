import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

import torch
# import whisper

from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES
from interactive import interactive_infer_image
import cv2
import matplotlib.pyplot as plt

class SEEM:
    def __init__(self,conf_files=None):
        '''
        build args
        '''
        current_path = os.path.dirname(os.path.abspath(__file__))
        if conf_files is None:
            conf_files = os.path.join(current_path,"configs/seem/focall_unicl_lang_demo.yaml")
        opt = load_opt_from_config_files([conf_files])
        opt = init_distributed(opt)

        # META DATA
        model_dir = os.path.join(current_path, "../../models/")

        if 'focalt' in conf_files:
            pretrained_pth = os.path.join(model_dir,"seem_focalt_v0.pt")
            if not os.path.exists(pretrained_pth):
                os.system("wget {}".format("https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focalt_v0.pt"))
        elif 'focal' in conf_files:
            pretrained_pth = os.path.join(model_dir,"seem_focall_v0.pt")
            if not os.path.exists(pretrained_pth):
                os.system("wget {}".format("https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v0.pt"))

        '''
        build model
        '''

        model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
        with torch.no_grad():
            model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)

        '''
        audio
        '''
        #do not use audio
        # audio = whisper.load_model("base")
        self.model = model

    @torch.no_grad()
    def inference(self,image, task, *args, **kwargs):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            if 'Video' in task:
                #raise error
                raise ValueError("not support")
            else:
                return interactive_infer_image(self.model, None, image, task, *args, **kwargs)
        

    def seg_img_with_text(self,img,text):
        #img: HxWx3 np array
        #text: the object you need to find
        #masks: the masks of the object, 0-1 array
        #result: visulize the result
        img = Image.fromarray(img)
        task = ["example","Text"]
        image = {}
        image['image'] = img
        image['mask'] = None
        masks,result = self.inference(image, task,reftxt = text)
        return masks,result


if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    seem = SEEM()
    image_path = os.path.join(current_path,'../../example/robot1_rgb.png')
    text_input = "table"
    img_input = cv2.imread(image_path)
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    
    masks,result = seem.seg_img_with_text(img_input,text_input)
    for mask in masks:
        print(mask.shape)
        plt.imshow(mask)
        plt.axis('off')  # Turn off axis numbers and ticks
        plt.show()
    plt.imshow(result)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()
