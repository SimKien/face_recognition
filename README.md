# face_recognition

This repo contains a simple face recognition in the video provided by an input camera.

New users can be added with the script generate_user_images.py. This uses a camera (the main camera of the current system) and by pressing the button "s" the current frame is used to identify a face and save this face to the known users.

These known users can now be detected in an input video. By executing recognize_faces.py, a camera (the main camera of the system) gives an input video and the faces in the video are highlighted and compared to the known faces. If a face is known, the corresponding name is provided too, if a face is not known, "Unknown" is provided.