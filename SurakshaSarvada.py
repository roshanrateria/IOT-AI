import tkinter as tk
from tkinter import filedialog, simpledialog
import face_recognition
import cv2
from PIL import Image, ImageTk
import numpy as np

obama_image = face_recognition.load_image_file(r"C:\Users\rater\Downloads\download.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

known_face_encodings = [obama_face_encoding]
known_face_names = ["Avneet Kaur"]

def add_photos():
    file_paths = filedialog.askopenfilenames(title='Select Known Faces')

    for file_path in file_paths:
        # Load the selected image
        image = Image.open(file_path)
        image.thumbnail((200, 200))  # Resize the image to a smaller size

        # Convert to RGB format for face recognition
        rgb_image = image.convert("RGB")
        rgb_image = ImageTk.PhotoImage(rgb_image)

        # Create a new window for entering the name
        name_window = tk.Toplevel(root)
        name_window.title("Enter Name for Known Face")

        label = tk.Label(name_window, image=rgb_image)
        label.image = rgb_image
        label.pack()

        def get_name():
            name = simpledialog.askstring("Enter Name", "Enter the name for the known face:")
            if name:
                # Encode and add the face to the known faces
                image_array = np.array(image)
                face_encoding = face_recognition.face_encodings(image_array)[0]
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)

            name_window.destroy()

        name_button = tk.Button(name_window, text="OK", command=get_name, bg="green", fg="white")
        name_button.pack()

def start_monitoring():
    video_capture = cv2.VideoCapture(0)
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

# Create a tkinter window
root = tk.Tk()
root.title("Face Recognition and Monitoring")

# Set the window size
root.geometry("400x300")

# Create buttons for adding photos and starting monitoring
add_photos_button = tk.Button(root, text="Add Known Faces", command=add_photos, bg="blue", fg="white", width=20)
start_monitoring_button = tk.Button(root, text="Start Monitoring", command=start_monitoring, bg="green", fg="white", width=20)

add_photos_button.pack(pady=10)
start_monitoring_button.pack(pady=10)

root.mainloop()
