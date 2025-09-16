# CollaBot: Vision-Language Guided Simultaneous Collaborative Manipulation

This is the official demo code for CollaBot, the first method that uses large language models and vision-language models to perform simultaneous collaborative manipulation using multi-robot system.

<p align="center">
  <img src="media/move_chair.gif" width="700"><br>
  3 robots move a table collaboratively
</p>


<p align="center">
  <img src="media/3robot.gif" width="700"><br>
  3 robots move a table collaboratively
</p>

## Guideline

<!-- [[Project Page]]() [[Paper]]()  -->

We mainly divide CollaBot into three parts:
1. Task Inference using VLMs
2. Collaborative Grasp Pose Generation
3. Motion Planning for Multiple Robots

<div align="center">
  <img src="media/framework.png" width="700">
</div>
In this repository, we provide the complete source code for CollaBot including these three parts. Specifically, we offer a demo that visualizes our methods using Open3D, even if you do not have access to the VLM APIs. We also provide the implementation of CollaBot using Gazebo.

We also provide the *appendix.pdf* for your reference, which contains a detailed description of how we handle motion planning with closed-chain constraints.

The repository is structured as follows:
- A brief description of this work
- Instructions on how to install the repository
- Demos for experimenting with our methods
- Code structure


## Setup Instructions
Firstly, clone this repo.

```Shell
mkdir -p collabot/src && cd collabot/src
git clone ...
```



Then, install necessary pkgs for this work.

- Create a conda environment:
```Shell
conda create -n collabot python=3.9
conda activate collabot
```

- Install a proper version of PyTorch. We are using torch==1.13.0+cu116 in this work.


- Install other dependencies:
```Shell
pip install -r requirements.txt
pip install -r requirements_SEEM.txt
```

If you find some problem install `mpi4py`, you can run `conda install mpi4py==3.1.5`. Then, continue to run `pip install -r requirements.txt`.

- Download seem_focall_v0.pt in [This Link](https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v0.pt) to install SEEM. Then, put it into `./collabot/models/`

- Download our trained LoGNet from [Google Drive](https://drive.google.com/file/d/1YkEzUa1G8f6yWOn-E3VEBbKtDKv1hvH_/view?usp=sharing). Then put it in `./collabot/scripts/grasping/trained_model/`

- Obtain an [OpenAI API](https://openai.com/blog/openai-api) key, and put it inside the `./collabot/prompts/api_chatgpt.json`. If you do not have an api, you can also try our demo with VLMs in `/collabot/scripts/playground_basic.ipynb`

Your json file should look like this.
```json
{
    "api_key": "",
    "api_base": ""
}
```
### Downloading Supporting Packages
To run our code in simulation and for visulization, you should have ROS installed on your computer. We encourage you using ROS Noetic. You need to download some necessary pkgs for using in [Google Drive](https://drive.google.com/file/d/1bK9hPQ6i3jFVAErbMZeUla7R5nztpMGx/view?usp=sharing). Then, put them under `src`.

Your file structure should look like this.
```shell
.
├── collabot
├── gazebo_plugins
├── media
├── model_description
├── README.md
├── requirements_SEEM.txt
├── requirements.txt
└── robot_gazebo
```
## Running Demo without API and ROS

Demo code is at `collabot/scripts/playground_basic.ipynb`. Instructions can be found in the notebook.

## Running Demo with API

Demo code is at `collabot/scripts/playground_API.ipynb`. We provide an example of using QWEN for example. It will be very easy to modify this example for GPT 4. Also, you can modify it to make it suitable for Gemini or other VLMs.


## Running Demo in Simulation


You should run the following command to compile supporting pkgs.
```shell
cd PATH_to_COLLABOT/collabot/
catkin_make
source devel/setup.bash
```

Then, run the following command to open the simulation in Gazebo
```shell
roslaunch robot_gazebo table_chair.launch
```

Demo code is at `collabot/scripts/playground_SIM.ipynb`. 


## Running in Your Environment
You can use Gazebo to create your own environment. Then, use `example_control.ipynb` to capture imgs of the scene.


## Code Structure

Files:

- **`modules`**:
  - **`collabot/scripts/grasping`**: collaborative grasping module.
  - **`collabot/scripts/SEEM`**: object detection using SEEM.
  - **`collabot/scripts/motion_planning`**: motion planning module.
- **`collabot/prompts`**: api and prompts.

Core to Collabot:

- **`playground_***.ipynb`**: Playground for CollaBot.
- **`create_object_grasp_pose.py`**: Implementation of large-object collaborative grasping.
- **`motion_planning.py`**: Implementation of motion planning in two stages.
- **`example_control.py`**: You can use this code to capture images from the env.




