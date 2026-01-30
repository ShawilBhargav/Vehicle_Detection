# Real-Time Vehicle Detection System (YOLOv3)

This project is a real-time vehicle detection system developed using Python and the YOLOv3 deep learning model. It is designed to detect vehicles from video input and display bounding boxes in real time. The system provides fast and accurate detection, making it suitable for traffic monitoring, surveillance applications, and academic projects related to computer vision and artificial intelligence.

To run this project, Python 3 and OpenCV must be installed on your system. You can install the required dependency using the command pip install opencv-python. Additional YOLO configuration files are already included in the project directory and are automatically loaded during execution.

Due to GitHub file size limitations, the YOLOv3 pre-trained weights file is not included in this repository. You must manually download the yolov3.weights file from Kaggle using the following link: https://www.kaggle.com/datasets/shivam316/yolov3-weights
. After downloading, place the file inside the yolo folder of the project directory so that the final path becomes yolo/yolov3.weights. The project will not run correctly if this file is missing or placed in the wrong directory.

Once the weights file is added, the project can be executed by running the main Python file using the command python main.py. After execution, the system will start processing the input video and display detected vehicles with bounding boxes in real time.

This project demonstrates the practical implementation of deep learning-based object detection and highlights the use of YOLO for real-time applications. It is suitable for learning purposes, portfolio projects, and demonstrations of computer vision capabilities.
