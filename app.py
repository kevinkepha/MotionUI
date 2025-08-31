"""
MotionUI — Real-Time Gesture Recognition Demo

- Uses MediaPipe Hands for hand detection & tracking
- Rule-based gesture classification:
    ✊ Fist
    🖐️ Open Palm
    ☝️ Pointing (index)
    👍 Thumbs Up
- Gradio UI supports webcam stream and image upload
Author: Duncan
"""

import cv2
import numpy as np
import mediapipe as mp
import gradio as gr
from typing import Tuple, Optional

# -----------------------
# MediaPipe setup (global for reuse)
# -----------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Keep a single Hands instance for realtime performance
# static_image_mode=False -> optimized for video / streaming
_hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)


# -----------------------
# Gesture classification utilities
# -----------------------
def _finger_is_up(landmarks, finger_tip_idx: int, finger_pip_idx: int) -> bool:
    """
    Simple test whether a finger is "up" by comparing y of tip vs pip.
    (In image coords y increases downward, so tip.y < pip.y => finger up)
    """
    return landmarks[finger_tip_idx].y < landmarks[finger_pip_idx].y


def _thumb_is_up(landmarks, handedness: str) -> bool:
    """
    Thumb up heuristics: check x coordinates relative to thumb MCP/IP depending on hand side.
    For right hand, thumb up (extended to right) -> tip.x > ip.x
    For left hand, thumb up (extended to left)  -> tip.x < ip.x
    """
    tip = landmarks[4]
    ip = landmarks[3]
    mcp = landmarks[2]
    if handedness == "Right":
        return tip.x > ip.x and tip.x > mcp.x
    else:
        return tip.x < ip.x and tip.x < mcp.x


def classify_gesture_from_landmarks(landmarks, handedness: str) -> str:
    """
    Classify basic gestures from a single hand's landmarks.
    Returns a short label string with an emoji.
    """
    # Finger indices: Thumb=4 tip, Index=8 tip, Middle=12, Ring=16, Pinky=20
    finger_states = [
        _finger_is_up(landmarks, 8, 6),   # index
        _finger_is_up(landmarks, 12, 10), # middle
        _finger_is_up(landmarks, 16, 14), # ring
        _finger_is_up(landmarks, 20, 18), # pinky
    ]
    thumb_up = _thumb_is_up(landmarks, handedness)

    # Evaluate common gestures
    # Fist: no fingers up and thumb not up
    if sum(finger_states) == 0 and not thumb_up:
        return "✊ Fist"

    # Open palm: all fingers up (and thumb approx up/out)
    if sum(finger_states) == 4 and thumb_up:
        return "🖐️ Open Palm"

    # Pointing: only index up
    if finger_states == [1, 0, 0, 0]:
        return "☝️ Pointing"

    # Thumbs up: thumb up, others down
    if thumb_up and sum(finger_states) == 0:
        return "👍 Thumbs Up"

    # Fallback
    return "🤷 Unknown Gesture"


# -----------------------
# Frame processing
# -----------------------
def annotate_frame_and_classify(frame_bgr: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Process a BGR frame (OpenCV), detect hands, draw landmarks,
    and return (annotated_rgb_frame, gesture_label).
    If multiple hands are detected, returns the label for the primary hand (first).
    """
    if frame_bgr is None:
        return None, "No Frame"

    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # To improve performance, mark the image as not writeable to pass by reference
    image_rgb.flags.writeable = False
    results = _hands.process(image_rgb)
    image_rgb.flags.writeable = True

    annotated = frame_bgr.copy()
    gesture_label = "No Hand"

    if results.multi_hand_landmarks:
        # If handedness info is available, choose the first hand's label
        handedness_list = []
        if results.multi_handedness:
            for h in results.multi_handedness:
                # h.classification is a list of classification with label and score
                handedness_list.append(h.classification[0].label)

        # iterate through hands and annotate
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw landmarks and connections on the annotated image
            mp_drawing.draw_landmarks(
                annotated,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=2),
            )

            # Convert normalized landmarks to simple list for classification
            lm = hand_landmarks.landmark
            handedness = handedness_list[idx] if idx < len(handedness_list) else "Right"

            # Classify this hand
            this_gesture = classify_gesture_from_landmarks(lm, handedness)

            # Use first detected hand as primary label (but show labels for both)
            if idx == 0:
                gesture_label = this_gesture

            # Put small label near wrist area (landmark 0)
            h, w = annotated.shape[:2]
            wrist = lm[0]
            x = int(wrist.x * w)
            y = int(wrist.y * h) - 20
            cv2.putText(
                annotated,
                this_gesture,
                (max(5, x), max(20, y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    # Overlay a primary gesture text on top-left
    cv2.rectangle(annotated, (0, 0), (360, 40), (0, 0, 0), -1)  # background strip
    cv2.putText(
        annotated,
        f"Gesture: {gesture_label}",
        (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Convert BGR->RGB for Gradio display
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated_rgb, gesture_label


# -----------------------
# Gradio functions
# -----------------------
def infer(image: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], str]:
    """
    Gradio wrapper: accepts an RGB image (numpy), returns (annotated_rgb, label).
    Gradio's Image type provides an RGB numpy array; convert to BGR for processing.
    """
    if image is None:
        return None, "No Input"

    # Convert RGB (gradio) to BGR (OpenCV)
    frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated_rgb, label = annotate_frame_and_classify(frame_bgr)
    return annotated_rgb, label


# -----------------------
# Gradio UI
# -----------------------
def build_ui():
    title = "MotionUI — Real-Time Gesture Recognition"
    description = """MotionUI recognizes common hand gestures in real-time using your webcam
or uploaded images. It demonstrates how touchless gesture interfaces can be integrated
into AR/VR, HCI, gaming, and assistive technologies."""
    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)

        with gr.Row():
            inp = gr.Image(source="webcam", tool="editor", type="numpy", label="Webcam / Upload")
            out_img = gr.Image(type="numpy", label="Annotated Output")
            out_label = gr.Textbox(label="Recognized Gesture", interactive=False)

        # Live processing: when the webcam updates, Gradio calls the function
        inp.change(fn=infer, inputs=[inp], outputs=[out_img, out_label], every=0.1)

        # Also provide a run button for upload mode (optional)
        run_btn = gr.Button("Run (Upload)")
        run_btn.click(fn=infer, inputs=[inp], outputs=[out_img, out_label])

        gr.Markdown("Built with **MediaPipe**, **OpenCV**, and **Gradio**. © Duncan")

    return demo


if __name__ == "__main__":
    app = build_ui()
    # launch with share=False by default. Set share=True to get a public link (useful for demos)
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
