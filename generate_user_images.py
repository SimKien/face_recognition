import cv2 as cv
import face_recognition

name = input("Enter your name: ")

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()

    cv.imshow("Save new user", frame)

    input = cv.waitKey(1)
    if input & 0xFF == ord("s"):
        face_locations = face_recognition.face_locations(frame)

        if len(face_locations) == 0:
            print("No face detected!")
            continue
        elif len(face_locations) > 1:
            print("Multiple faces detected!")
            continue
        frame = frame[
            face_locations[0][0] : face_locations[0][2],
            face_locations[0][3] : face_locations[0][1],
        ]
        cv.imwrite(f"known_users/{name}.png", frame)
        break

    if input & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
