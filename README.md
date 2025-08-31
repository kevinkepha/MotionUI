# 🖐️ MotionUI: RealTime Gesture Recognition for Touchless Interaction

**MotionUI** is a computer vision application that enables **realtime hand gesture recognition** using a standard webcam.  
It demonstrates how touchless interaction can power **AR/VR systems, humancomputer interaction (HCI), gaming, and assistive technologies**.  

Built with **MediaPipe**, **OpenCV**, and **Gradio**, MotionUI offers an intuitive demo and an extensible foundation for integration into realworld products.



## ✨ Features
 Detects and tracks hands in real time.  
 Recognizes common gestures:
   ✊ Fist  
   🖐️ Open Palm  
   ☝️ Pointing  
   👍 Thumbs Up  
 Provides a clean **web interface** powered by Gradio.  
 Supports both **image uploads** and **live webcam feed**.  
 Designed for extension into **AR/VR apps, kiosks, and touchless systems**.  



## 📂 Project Structure
```

motionui/
│── app.py              # Main application (Gradio demo)
│── requirements.txt    # Dependencies
│── README.md           # Project documentation

````



## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<yourusername>/motionui.git
cd motionui
python m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install r requirements.txt
````



## ▶️ Usage

Run the app:

```bash
python app.py
```

Gradio will launch a local server. Open the link in your browser to start using **MotionUI**.

* Use **webcam mode** to test realtime gestures.
* Or upload an image containing a hand.



## 🔧 Extensibility

* Add more gestures (👌 OK sign, ✌️ Victory, 🤘 Rock).
* Integrate with **keyboard/mouse controls** (e.g., PowerPoint navigation).
* Export recognized gestures to **VR/AR apps** or **IoT devices**.
* Replace rulebased recognition with a **trained ML classifier** for robust detection.



## 🖼️ Demo

(Include screenshots or a GIF of the app in action once you record it.)



## 📝 License

MIT — free to use and modify.





