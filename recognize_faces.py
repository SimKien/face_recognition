import os
import cv2 as cv
import numpy as np
import face_recognition

PROCESS_EVERY_NTH_FRAME = 10

KNOWN_USERS_DIRECTORY = "known_users"

known_face_encodings = []
known_face_names = []

filenames = os.listdir(KNOWN_USERS_DIRECTORY)
filenames = [filename for filename in filenames if filename.endswith(".png")]

if len(filenames) == 0:
    print("No known users found in 'known_users' directory.")
    exit(1)

for filename in filenames:
    image = face_recognition.load_image_file(f"{KNOWN_USERS_DIRECTORY}/{filename}")
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(os.path.splitext(filename)[0])

video_capture = cv.VideoCapture(0)

face_locations = []
face_names: list[str] = []
last_processed = 0
process_this_frame = True

while True:
    ret, frame = video_capture.read()

    if process_this_frame:
        current_face_encodings = []
        face_names.clear()

        small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]
        rgb_small_frame = rgb_small_frame.astype(np.uint8)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        current_face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        for face_encoding in current_face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding
            )
            name = "Unknown"

            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv.rectangle(
            frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv.FILLED
        )

        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv.imshow("Detection", frame)

    # only process every nth frame
    last_processed += 1
    if last_processed >= PROCESS_EVERY_NTH_FRAME:
        last_processed = 0
    process_this_frame = last_processed == 0

    # if the `q` key is pressed, break from the loop
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv.destroyAllWindows()
