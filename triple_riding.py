import cv2
from ultralytics import YOLO

# Load Triple Riding YOLO Model
model = YOLO(
    r"C:\Users\Admin\Desktop\TrinetraAI\ANPR-and-ATCC-for-Smart-Traffic-Management\triple riding.pt"
)

# Class IDs (as per your trained model)
motorbike_class = 3
person_class = 0


def detect_triple_riding(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # YOLO prediction
        results = model.predict(source=frame, conf=0.5, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()

        motorbikes = []
        people = []

        # Split detections
        for det in detections:
            x1, y1, x2, y2, score, class_id = det
            class_id = int(class_id)

            if class_id == motorbike_class:
                motorbikes.append([x1, y1, x2, y2])
            elif class_id == person_class:
                people.append([x1, y1, x2, y2])

        # Check triple riding
        for i, bike in enumerate(motorbikes, start=1):
            x1b, y1b, x2b, y2b = bike
            person_count = 0

            for person in people:
                x1p, y1p, x2p, y2p = person

                # Overlap check
                if (
                    x1p < x2b and x2p > x1b and
                    y1p < y2b and y2p > y1b
                ):
                    person_count += 1

            status = "Offense" if person_count >= 3 else "Not Offense"
            color = (0, 0, 255) if person_count >= 3 else (0, 255, 0)

            cv2.rectangle(frame, (int(x1b), int(y1b)), (int(x2b), int(y2b)), color, 2)
            cv2.putText(
                frame,
                f"{person_count} - {status}",
                (int(x1b), int(y1b) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        # Show output
        cv2.imshow("Triple Riding Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video1 = r"C:\Users\Admin\Desktop\TrinetraAI\ANPR-and-ATCC-for-Smart-Traffic-Management\sample detection videos\triple riding.mp4"
    video2 = r"C:\Users\Admin\Desktop\TrinetraAI\ANPR-and-ATCC-for-Smart-Traffic-Management\sample detection videos\triple riding 2.mp4"

    print("Running Triple Riding Detection...")
    detect_triple_riding(video1)
    detect_triple_riding(video2)
