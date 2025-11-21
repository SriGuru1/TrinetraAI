import cv2
import numpy as np
from ultralytics import YOLO
import os

# Function to simulate a traffic light on the frame
def simulate_traffic_light(frame, light_state, position):
    traffic_light_colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0)]
    x, y = position

    cv2.rectangle(frame, (x, y), (x + 50, y + 150), (50, 50, 50), -1)

    for i, color in enumerate(traffic_light_colors):
        cv2.circle(frame, (x + 25, y + 25 + i * 50),
                   15, color if i == light_state else (50, 50, 50), -1)

# Determine signal color based on traffic
def determine_signal(total_vehicles):
    if total_vehicles < 10:
        return "Green", (0, 255, 0), 2
    elif total_vehicles < 20:
        return "Yellow", (0, 255, 255), 1
    else:
        return "Red", (0, 0, 255), 0

# Process each frame
def process_frame(frame, model, road_name, directions):
    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()
    vehicle_count = {"car": 0, "truck": 0, "motorcycle": 0, "bus": 0}

    for det in detections:
        x1, y1, x2, y2, conf, class_id = det
        class_id = int(class_id)

        if class_id in model.names:
            label = model.names[class_id]
        else:
            continue

        if label in vehicle_count:
            vehicle_count[label] += 1

            center_x = (x1 + x2) / 2
            if center_x < frame.shape[1] / 2:
                directions['left'] += 1
            else:
                directions['right'] += 1

            color = (0, 255, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(frame, f"Road: {road_name}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    y_offset = 50
    for direction, count in directions.items():
        cv2.putText(frame, f"{direction.capitalize()}: {count}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        y_offset += 25

    total_vehicles = sum(vehicle_count.values())

    cv2.putText(frame, f"Total Vehicles: {total_vehicles}",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
    y_offset += 25

    signal_color, signal_rgb, light_state = determine_signal(total_vehicles)

    cv2.putText(frame, f"Signal: {signal_color}",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                signal_rgb, 2)

    simulate_traffic_light(frame, light_state, position=(frame.shape[1]-100, 20))

    return frame

# Main function to process videos
def process_atcc_videos(input_files, model):
    caps = []
    road_names = []

    for file in input_files:
        cap = cv2.VideoCapture(file)
        if not cap.isOpened():
            print(f"Error: Cannot open {file}")
            continue
        caps.append(cap)
        road_names.append(os.path.basename(file))

    target_width, target_height = 640, 480

    while True:
        frames = []
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.resize(frame, (target_width, target_height))
            directions = {"left": 0, "right": 0}
            processed = process_frame(frame, model, road_names[i], directions)
            frames.append(processed)

        if len(frames) == 0:
            break

        rows = []
        for i in range(0, len(frames), 2):
            if i + 1 < len(frames):
                rows.append(np.hstack((frames[i], frames[i+1])))
            else:
                rows.append(frames[i])

        grid = np.vstack(rows)
        cv2.imshow("ATCC System", grid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()


# ------------------------ RUN HERE ------------------------

if __name__ == "__main__":
    model = YOLO(
        r"C:\Users\Admin\Desktop\TrinetraAI\ANPR-and-ATCC-for-Smart-Traffic-Management\atcc.pt"
    )

    input_files = [
        r"C:\Users\Admin\Desktop\TrinetraAI\ANPR-and-ATCC-for-Smart-Traffic-Management\sample detection videos\atcc sample.mp4"
    ]

    process_atcc_videos(input_files, model)
