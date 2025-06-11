import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import os
import datetime
import numpy as np

# Paths to the ResNet SSD model
MODEL_PATH = "SpySee/dlib-19.24.99-cp312-cp312-win_amd64.whl"
PROTOTXT_PATH = "D:/main spysee/SpySee/deploy.prototxt"


# Load the ResNet SSD model
net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

# Directory for saving images
downloads_directory = os.path.join(os.path.expanduser("~"), "Downloads", "Face_Images")
os.makedirs(downloads_directory, exist_ok=True)

def login():
    login_id = login_id_entry.get()
    password = password_entry.get()

    if login_id == "spysee" and password == "spysee":
        messagebox.showinfo("Login Successful", "Welcome, Spysee!")
        root.withdraw()
        start_face_detection()
    else:
        messagebox.showerror("Login Failed", "Invalid login credentials!")

def compute_iou(boxA, boxB):
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def start_face_detection():
    known_faces = {
        "Karthik": "karthik1.jpg",
       # "Karthik": "C:/Users/DELL/OneDrive/Desktop/SpySee1/karthik1.jpg",
        #"Karthik": "C:/Users/DELL/OneDrive/Desktop/SpySee1/karthik3.jpg",
        #"Karthik": "C:/Users/DELL/OneDrive/Desktop/SpySee1/karthik copy.jpg",
        #"Iqbal": "C:/Users/DELL/OneDrive/Desktop/SpySee1/iqbal1.jpg"

        
    }
    known_face_encodings = {}

    # Preload known face encodings
    for name, image_path in known_faces.items():
        image = cv2.imread(image_path)
        blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        if detections.shape[2] > 0:
            known_face_encodings[name] = detections[0, 0, 0, 3:7]  # Coordinates for face location

    cap = cv2.VideoCapture(0)

    video_filename = f"face_video_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.avi"
    video_path = os.path.join(downloads_directory, video_filename)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width, frame_height))

    def display_frame():
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()

            face_identifications = []

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Confidence threshold
                    box = detections[0, 0, i, 3:7] * [w, h, w, h]
                    x1, y1, x2, y2 = box.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)  # Ensure valid coordinates
                    x2, y2 = min(w, x2), min(h, y2)

                    face = frame[y1:y2, x1:x2]

                    if face.size > 0:  # Ensure face region is not empty
                        name = "Unknown"
                        max_iou = 0

                        # Match with known faces using IoU
                        for known_name, known_encoding in known_face_encodings.items():
                            known_box = (known_encoding * [w, h, w, h]).astype(int)
                            iou = compute_iou((x1, y1, x2, y2), tuple(known_box))
                            if iou > 0.3 and iou > max_iou:  # IoU threshold
                                name = known_name
                                max_iou = iou

                        face_identifications.append(name)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        if name == "Unknown":
                            img_name = f"face_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                            img_path = os.path.join(downloads_directory, img_name)
                            cv2.imwrite(img_path, face)
                            out.write(frame)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(image=img)

            label.imgtk = img_tk
            label.config(image=img_tk)

        label.after(10, display_frame)

    face_detection_window = tk.Toplevel()
    face_detection_window.title("Face Detection")

    label = tk.Label(face_detection_window)
    label.pack(expand=True, fill=tk.BOTH)

    display_frame()

    def close_camera():
        cap.release()
        out.release()
        face_detection_window.destroy()

    face_detection_window.protocol("WM_DELETE_WINDOW", close_camera)

def show_about_us():
    messagebox.showinfo("About Us", "This application is developed by Spysee Inc.")

def show_contact():
    messagebox.showinfo("Contact", "For support, contact support@spysee.com")

def update_video():
    ret, frame = video_capture.read()
    if ret:
        frame_resized = cv2.resize(frame, (root.winfo_width(), root.winfo_height()))
        frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_resized)
        frame_photo = ImageTk.PhotoImage(frame_image)
        video_label.config(image=frame_photo)
        video_label.image = frame_photo
    video_label.after(1, update_video)

root = tk.Tk()
root.title("Login Page")

# Initialize the video background
video_path = "C:/Users/DELL/OneDrive/Desktop/SpySee1/SpySee/video.mp4"
video_capture = cv2.VideoCapture(video_path)

video_label = tk.Label(root)
video_label.place(x=0, y=0, relwidth=1, relheight=1)
update_video()

# Set window dimensions
window_width = 600
window_height = 650  # Increase height to accommodate more space for buttons
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
position_top = int(screen_height / 2 - window_height / 2)
position_right = int(screen_width / 2 - window_width / 2)
root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')


# Main frame for login
login_frame = tk.Frame(root, bg='#babab6', bd=10, relief='ridge')
login_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Heading label
heading_label = tk.Label(login_frame, text="SPYSEE", bg='#babab6', fg='#050505', font=("Arial", 18, "bold"))
heading_label.grid(row=0, columnspan=2, pady=(10, 20))

# Login ID label and entry
login_id_label = tk.Label(login_frame, text="Login ID:", bg='#babab6', fg='#050505', font=("Arial", 12))
login_id_label.grid(row=1, column=0, pady=5, padx=5)
login_id_entry = tk.Entry(login_frame, font=("Arial", 12))
login_id_entry.grid(row=1, column=1, pady=5, padx=5)

# Password label and entry
password_label = tk.Label(login_frame, text="Password:", bg='#babab6', fg='#050505', font=("Arial", 12))
password_label.grid(row=2, column=0, pady=5, padx=5)
password_entry = tk.Entry(login_frame, show="*", font=("Arial", 12))
password_entry.grid(row=2, column=1, pady=5, padx=5)

# Login button
login_button = tk.Button(login_frame, text="Login", command=login, bg='#FFA500', fg='white', font=("Arial", 12), relief='raised')
login_button.grid(row=3, columnspan=2, pady=10)

# About Us button
about_us_button = tk.Button(login_frame, text="About Us", command=show_about_us, bg='#FFA500', fg='white', font=("Arial", 12), relief='raised')
about_us_button.grid(row=4, columnspan=2, pady=5)

# Contact button
contact_button = tk.Button(login_frame, text="Contact Us", command=show_contact, bg='#FFA500', fg='white', font=("Arial", 12), relief='raised')
contact_button.grid(row=5, columnspan=2, pady=5)

root.mainloop()


