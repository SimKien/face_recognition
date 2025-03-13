import os
import cv2 as cv
from deepface import DeepFace

RECOGNIZE_EVERY_N_FRAMES = 5

KNOWN_USERS_DIRECTORY = "known_users"

folder = os.path.abspath(KNOWN_USERS_DIRECTORY)

cap = cv.VideoCapture(0)

counter = 0
process_detection = True

face_locations: list[tuple[int, int, int, int]] = []
face_names: list[str] = []

while True:
    ret, frame = cap.read()

    if process_detection:
        small_frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)

        face_locations.clear()
        face_names.clear()

        faces = DeepFace.extract_faces(small_frame, enforce_detection=False)

        for face in faces:
            face_pic = face["face"]
            area = face["facial_area"]
            x, y, w, h = area["x"], area["y"], area["w"], area["h"]
            x = x * 2
            y = y * 2
            w = w * 2
            h = h * 2
            face_locations.append((x, y, x + w, y + h))

            result = DeepFace.find(
                face_pic,
                db_path=folder,
                model_name="VGG-Face",
                distance_metric="cosine",
                enforce_detection=False,
                silent=True,
            )

            if len(result) == 0:
                face_names.append("Unknown")
                continue
            if result[0].empty:
                face_names.append("Unknown")
                continue

            path = result[0]["identity"].values[0]
            name = os.path.basename(path)
            name = os.path.splitext(name)[0]
            face_names.append(name)

    for name, (left, top, right, bottom) in zip(face_names, face_locations):
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv.putText(
            frame, name, (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

    cv.imshow("Detect user", frame)

    counter += 1
    if counter == RECOGNIZE_EVERY_N_FRAMES:
        counter = 0
        process_detection = True
    else:
        process_detection = False

    input = cv.waitKey(1)
    if input & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
