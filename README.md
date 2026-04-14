**📌 Face Detection Attendance System**


**📖 Overview**

The Face Detection Attendance System is an AI-based application that automates attendance marking using face detection and recognition techniques. The system captures real-time video through a webcam, detects faces, recognizes registered users, and automatically records attendance in a database. This project eliminates manual effort, reduces errors, and prevents proxy attendance  by using unique facial features for identification.

**🎯 Features**
1.Real-time face detection using webcam
2.Face recognition of registered users
3.Automatic attendance marking with date and time
4.Prevents duplicate entries
5.User-friendly interface
6.Secure and contactless system

**🛠️ Technologies Used**
1.Programming Language: Python
2.Libraries: OpenCV, NumPy, Pandas
3.Framework: Flask (for web interface)
4.Database: MySQL / CSV
5.Concepts: Computer Vision, Deep Learning

**⚙️ Algorithms Used**
1.Face Detection: Haar Cascade Classifier
2.Face Recognition: LBPH (Local Binary Pattern Histogram)

**📂 Project Structure**
Face-Detection-Attendance-System/
│
├── dataset/                # Stored images of registered users
├── trainer/                # Trained model files
├── static/                 # CSS, JS, images
├── templates/              # HTML files (UI)
├── app.py                  # Main application file
├── attendance.csv          # Attendance records
├── requirements.txt        # Dependencies
└── README.md               # Project documentation

**🚀 How It Works**
Start the application and activate the webcam
Detect faces in real-time using OpenCV
Recognize faces using trained model
Match with stored dataset
Automatically mark attendance in database

**💻 Installation & Setup**
1️⃣ Clone the Repository
git clone https://github.com/your-username/face-attendance-system.git
cd face-attendance-system

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Application
python app.py

**📊 Output**
1.Recognized face displayed on screen
2.Attendance recorded with timestamp
3.Stored in CSV / database
