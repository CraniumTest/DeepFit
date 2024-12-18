import cv2
import numpy as np
import mediapipe as mp

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

    def estimate_pose(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        return results

    def draw_landmarks(self, frame, results):
        self.mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return frame

    def analyze_motion(self, results):
        # Pseudo-analysis: This should be replaced with a more detailed motion analysis
        landmarks = results.pose_landmarks

        if not landmarks: return "No landmarks detected"

        # Example: Check if the right elbow is bent
        right_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        if right_elbow.visibility < 0.5:
            return "Move detected but may be occluded or not clear."

        # Return additional analysis here
        return "Detected movement with landmark visibility."

    def release(self):
        self.pose.close()

def main():
    cap = cv2.VideoCapture(0)  # Use webcam or replace with a video file path

    pose_estimator = PoseEstimator()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = pose_estimator.estimate_pose(frame)

        frame = pose_estimator.draw_landmarks(frame, results)

        motion_feedback = pose_estimator.analyze_motion(results)
        cv2.putText(frame, motion_feedback, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('DeepFit Motion Analysis', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # Exit on ESC key
            break

    cap.release()
    cv2.destroyAllWindows()
    pose_estimator.release()

if __name__ == "__main__":
    main()
