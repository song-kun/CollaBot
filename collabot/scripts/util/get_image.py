import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class get_image():
    def __init__(self,robot_name_list):
        self.robot_name_list = robot_name_list
        self.rgb_image_dict = {}
        self.depth_image_dict = {}
        self.bridge = CvBridge()  # For converting ROS Image messages to OpenCV format

        self.rgb_topic_list = [f"/{robot_name}_d435/color/image_raw" for robot_name in robot_name_list]
        self.depth_topic_list = [f"/{robot_name}_d435/depth/image_raw" for robot_name in robot_name_list]


        # Initialize subscribers for each robot's RGB and depth topics
        for i, (rgb_topic, depth_topic) in enumerate(zip(self.rgb_topic_list, self.depth_topic_list)):
            rospy.Subscriber(rgb_topic, Image, self.rgb_callback, callback_args=robot_name_list[i])
            rospy.Subscriber(depth_topic, Image, self.depth_callback, callback_args=robot_name_list[i])

    def rgb_callback(self, msg, robot_name):
        # Convert the ROS Image message to an OpenCV image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.rgb_image_dict[robot_name] = cv_image  # Store the image in the dictionary
        except Exception as e:
            rospy.logerr(f"Failed to process RGB image for {robot_name}: {e}")

    def depth_callback(self, msg, robot_name):
        # Convert the ROS Image message to an OpenCV depth image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            self.depth_image_dict[robot_name] = cv_image  # Store the depth image in the dictionary
        except Exception as e:
            rospy.logerr(f"Failed to process depth image for {robot_name}: {e}")

    def write_image(self, robot_name,img_path = './',postfix = ''):
        # Example method to save the images for a specific robot to disk
        if robot_name in self.robot_name_list:
            if self.rgb_image_dict is None or self.depth_image_dict is None:
                rospy.logerr("No images received yet")
                return
            rgb_image = self.rgb_image_dict[robot_name]
            depth_image = self.depth_image_dict.get(robot_name)
            if rgb_image is not None:
                cv2.imwrite(f"{img_path}{robot_name}_rgb{postfix}.png", rgb_image)
                print("success save image")
            else:
                print(f"Failed to save RGB image for {robot_name}")
            if depth_image is not None:
                cv2.imwrite(f"{img_path}{robot_name}_depth{postfix}.png", depth_image)