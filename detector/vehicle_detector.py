import cv2
import math
import os

# ---------------- PATH SETUP ----------------

base_path = os.path.dirname(os.path.abspath(__file__))

weights_path = os.path.join(base_path, "..", "yolo", "yolov3.weights")
config_path = os.path.join(base_path, "..", "yolo", "yolov3.cfg")
coco_path = os.path.join(base_path, "..", "yolo", "coco.names")

# ---------------- LOAD CLASSES ----------------

classes = []
with open(coco_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

vehicle_classes = ["car", "motorbike", "bus", "truck"]

# ---------------- LOAD YOLO ----------------

net = cv2.dnn.readNet(weights_path, config_path)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

print("YOLO Loaded Successfully")

# ---------------- TRACKING MEMORY ----------------

tracked_vehicles = {}
vehicle_id_counter = 0

# ---------------- DISTANCE FUNCTION ----------------

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# ---------------- MAIN DETECTION FUNCTION ----------------

def detect_vehicles(frame):

    global tracked_vehicles, vehicle_id_counter

    height, width, _ = frame.shape

    blob = cv2.dnn.blobFromImage(
        frame,
        1 / 255,
        (320, 320),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []
    current_centers = []

    # ----------- DETECTION ------------

    for output in outputs:
        for detection in output:

            scores = detection[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]

            class_name = classes[class_id]

            if confidence > 0.6 and class_name in vehicle_classes:

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # ----------- NMS ------------

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.6)

    if len(indexes) > 0:

        for i in indexes.flatten():

            x, y, w, h = boxes[i]

            cx = x + w // 2
            cy = y + h // 2

            current_centers.append((cx, cy))

            label = classes[class_ids[i]]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ----------- CENTROID TRACKING ------------

    new_tracked = {}

    for center in current_centers:

        matched = False

        for vid, prev_center in tracked_vehicles.items():

            dist = calculate_distance(center, prev_center)

            if dist < 50:
                new_tracked[vid] = center
                matched = True
                break

        if not matched:
            vehicle_id_counter += 1
            new_tracked[vehicle_id_counter] = center

    tracked_vehicles = new_tracked

    total_vehicles = vehicle_id_counter

    return frame, total_vehicles

# if __name__ == "__main__":
#     cap = cv2.VideoCapture("../videos/traffic.mp4")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         frame,total,counts = detect_vehicles(frame)
#         cv2.imshow("Test",frame)
#         if cv2.waitKey(1)==27:
#             break
#     cap.release()
#     cv2.destroyAllWindows()