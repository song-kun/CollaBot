# import openai
from openai import OpenAI

import ast
import os
from PIL import Image
import base64
import json
import matplotlib.pyplot as plt
import numpy as np
import io
import re

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class prompt_interface:
    def __init__(self, api_key=None, api_base=None):
        if api_key is None:
            print("Please input the api_key")

        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base if api_base is not None else None,
        )
        self.target_objects = None

    def base_interface(self, img, prompt, model="qwen-vl-plus"):
        image_pil = Image.fromarray(img)

        buffered = io.BytesIO()
        image_pil.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()

        base64_image = base64.b64encode(img_bytes).decode("utf-8")

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }],
                stream=True
            )


            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
            
            
            return full_response


        except Exception as e:
            print(f"Wrong: {e}")
            return

    def object_detect(self, instruction):
        try:
            response = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[{
                    "role": "user",
                    "content": "List all objects in this sentence, return the python-list struct with the type of string, without any other words:" + instruction
                }]
            )
            x = response.choices[0].message.content
            lst = ast.literal_eval(x)
            return lst
        except Exception as e:
            print(f"Wrong: {e}")
            return

    def image_object_detection(self, image_path, target_object):
        base64_image = self.encode_image(image_path)
        object_name_str = ",".join(target_object)
        try:
            response = self.client.chat.completions.create(
                model="qwen-vl-plus",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"List all entities in the scene in the picture. You should select among {object_name_str}. Return the python-list struct with the type of string, without any other words."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }]
            )
            x = response.choices[0].message.content
            lst = ast.literal_eval(x)
            return lst
        except Exception as e:
            print(f"Wrong: {e}")
            return

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            img_bytes = image_file.read()
        return base64.b64encode(img_bytes).decode("utf-8")
    
    def obtain_constraint(self,img_rgb,prompt, instruction, max_robot_num=3,model="qwen-vl-plus"):
        
        values = {
            "max_num": max_robot_num,
            "task_description": instruction,
        }
        def replace_var(match):
            var_name = match.group(1)
            return str(values.get(var_name, match.group(0))) 

        modifed_prompt = re.sub(r"\$\((.*?)\)", replace_var, prompt)

        
        respones = self.base_interface(img_rgb,modifed_prompt,model)
        match = re.search(r'Constraint IS:\s*(.*)', respones)

        if match:
            return respones,match.group(1)
        else:
            return respones,None
        
    def choose_robot_num(self,img_rgb,prompt, instruction, max_robot_num=3,model="qwen-vl-plus"):
        
        values = {
            "max_num": max_robot_num,
            "task_description": instruction,
        }
        def replace_var(match):
            var_name = match.group(1)
            return str(values.get(var_name, match.group(0))) 

        modifed_prompt = re.sub(r"\$\((.*?)\)", replace_var, prompt)

        
        respones = self.base_interface(img_rgb,modifed_prompt,model)
        print(respones)
        match = re.search(r'NUMBER ROBOTS NEEDED IS:\s*(\d+)', respones)
        number_needed = -1
        if match:
            number_needed = int(match.group(1))
        
        return number_needed
        
    
if __name__ == "__main__":
    from PIL import Image

    current_path = os.path.dirname(os.path.abspath(__file__))
    api_file_path = os.path.join(current_path, '../../prompts/api_qwen.json')
    with open(api_file_path, "r") as f:
        config = json.load(f)

    api_key = config["api_key"]
    api_base = config["api_base"]


    instruction = "move the chair near the table"
    prompt_inter = prompt_interface(api_key,api_base)

    current_path = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_path,'../../example/robot1_rgb.png')
    
    # print(f"Given input prompt: {instruction}")
    # print("detect relative object in this object:",prompt_inter.object_detect(instruction))
    # all_object = prompt_inter.object_detect(instruction)
    # print("detected object in the scene",prompt_inter.image_object_detection(image_path,all_object))

    instruction = "move the table out of this room"
    color = np.array(Image.open(os.path.join(image_path)) )
    prompt_path = os.path.join(current_path, '../../prompts/choose_robot_num_prompts.txt')
    with open(prompt_path, "r") as f:
        prompt = f.read()
    respones = prompt_inter.choose_robot_num(color, prompt, instruction, max_robot_num=3, model="qwen-vl-plus")
    print(respones)
    