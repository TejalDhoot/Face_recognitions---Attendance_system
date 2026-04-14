from flask import Flask, render_template, Response, request, redirect
import cv2
import numpy as np
import os
import csv
from datetime import datetime

app = Flask(__name__)

# ---------- LOAD MODEL ----------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("model.yml")

label_map = np.load("labels.npy", allow_pickle=True).item()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

# ---------- ATTENDANCE ----------
def mark_attendance(name):
    file = "attendance.csv"
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    if not os.path.exists(file):
        with open(file, "w", newline="") as f:
            csv.writer(f).writerow(["Name","Date","Time"])

    with open(file, "r") as f:
        if any(name in line and date in line for line in f.readlines()):
            return

    with open(file, "a", newline="") as f:
        csv.writer(f).writerow([name,date,time])

# ---------- HOME ----------
@app.route('/')
def index():
    total = 0
    if os.path.exists("attendance.csv"):
        total = len(open("attendance.csv").readlines()) - 1
    return render_template("index.html", total=total)

# ---------- ADD STUDENT (FRAME + SPACE CAPTURE) ----------
@app.route('/add_student', methods=['GET','POST'])
def add_student():
    if request.method == 'POST':
        name = request.form['name'].strip()

        if name == "":
            return "❌ Name required"

        path = f"dataset/{name}"
        os.makedirs(path, exist_ok=True)

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces = face_cascade.detectMultiScale(gray, 1.2, 5)

            # 🔥 Instruction UI
            cv2.putText(frame, "Align face & press SPACE",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,255), 2)

            for (x, y, w, h) in faces:
                # 🔥 FACE FRAME
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))

                key = cv2.waitKey(1)

                # 🔥 PRESS SPACE TO CAPTURE
                if key == 32:
                    cv2.imwrite(f"{path}/0.jpg", face)

                    cv2.putText(frame, "Captured!",
                                (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0,255,0), 2)

                    cv2.imshow("Capture", frame)
                    cv2.waitKey(1500)

                    cap.release()
                    cv2.destroyAllWindows()

                    os.system("python train.py")

                    return redirect('/')

            cv2.imshow("Add Student Webcam", frame)

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        return redirect('/')

    return render_template("add_student.html")

# ---------- IMAGE UPLOAD ----------
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']

    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        result = "No face found"

        for (x, y, w, h) in faces:
            face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            label, conf = recognizer.predict(face)

            if conf < 80:
                name = label_map[label]
                result = f"Matched: {name}"
                mark_attendance(name)
            else:
                result = "Unknown"

        return render_template("index.html", prediction=result)

    return redirect('/')

# ---------- WEBCAM ----------
def generate_frames():
    while True:
        _, frame = camera.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            label, conf = recognizer.predict(face)

            if conf < 80:
                name = label_map[label]
                color = (0,255,0)
                mark_attendance(name)
            else:
                name = "Unknown"
                color = (0,0,255)

            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, name, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield(b'--frame\r\nContent-Type:image/jpeg\r\n\r\n'+buffer.tobytes()+b'\r\n')

@app.route('/webcam')
def webcam():
    return render_template("webcam.html")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------- ATTENDANCE ----------
@app.route('/attendance', methods=['GET','POST'])
def attendance():
    data = []

    if os.path.exists("attendance.csv"):
        rows = list(csv.reader(open("attendance.csv")))

        if request.method == 'POST':
            date = request.form['date']
            data = [r for r in rows if date in r]
        else:
            data = rows

    return render_template("attendance.html", data=data)

# ---------- RUN ----------
if __name__ == "__main__":
    app.run(debug=True)