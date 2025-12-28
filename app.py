import os
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# ==============================
# Load Colorization Model (SAFE PATHS)
# ==============================
def load_colorization_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models")

    proto = os.path.join(model_dir, "colorization_deploy_v2.prototxt")
    model = os.path.join(model_dir, "colorization_release_v2.caffemodel")
    pts_path = os.path.join(model_dir, "pts_in_hull.npy")

    print("Loading model files:")
    print(proto)
    print(model)
    print(pts_path)

    net = cv2.dnn.readNetFromCaffe(proto, model)

    pts = np.load(pts_path)
    pts = pts.transpose().reshape(2, 313, 1, 1)

    net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype(np.float32)]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [
        np.full([1, 313], 2.606, dtype="float32")
    ]

    return net


net = load_colorization_model()
current_model = "CNN"

# ==============================
# Webcam Capture
# ==============================
cap = cv2.VideoCapture(0)

# ==============================
# Colorization Function
# ==============================
def process_frame(frame):
    if current_model == "GRAY":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    L_resized = cv2.resize(L, (224, 224))
    L_resized = L_resized.astype("float32") - 50

    net.setInput(cv2.dnn.blobFromImage(L_resized))
    ab = net.forward()[0].transpose((1, 2, 0))
    ab = cv2.resize(ab, (frame.shape[1], frame.shape[0]))

    lab_out = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(lab_out.astype("float32"), cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    return (colorized * 255).astype("uint8")

# ==============================
# GUI Update Loop
# ==============================
def update():
    ret, frame = cap.read()
    if not ret:
        root.after(10, update)
        return

    frame = cv2.flip(frame, 1)
    output = process_frame(frame)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(output)
    img = ImageTk.PhotoImage(img)

    video_label.config(image=img)
    video_label.image = img

    root.after(10, update)

# ==============================
# Model Switch
# ==============================
def set_model(model):
    global current_model
    current_model = model

# ==============================
# GUI Setup
# ==============================
root = tk.Tk()
root.title("Real-Time Video Colorization")

video_label = tk.Label(root)
video_label.pack()

btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

tk.Button(
    btn_frame, text="CNN Colorization",
    command=lambda: set_model("CNN"),
    width=20
).grid(row=0, column=0, padx=10)

tk.Button(
    btn_frame, text="Grayscale",
    command=lambda: set_model("GRAY"),
    width=20
).grid(row=0, column=1, padx=10)

update()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
