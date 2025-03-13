import cv2 as cv
from deepface import DeepFace

name = input("Enter your name: ")

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()

    cv.imshow("Save new user", frame)

    input = cv.waitKey(1)
    if input & 0xFF == ord("s"):
        faces = DeepFace.extract_faces(frame, enforce_detection=False)

        if len(faces) == 0:
            print("No face detected")
            continue
        elif len(faces) > 1:
            print("More than one face detected")
            continue

        x, y, w, h = (
            faces[0]["facial_area"]["x"],
            faces[0]["facial_area"]["y"],
            faces[0]["facial_area"]["w"],
            faces[0]["facial_area"]["h"],
        )
        face = frame[y : y + h, x : x + w]
        cv.imwrite(f"known_users/{name}.jpg", face)
        break

    if input & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
