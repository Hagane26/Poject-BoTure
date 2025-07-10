import flet as ft
import time
import cv2
import math as m
import mediapipe as mp
import threading

# Mediapipe initialisasi
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def splash_screen(page: ft.Page):
    page.title = "Posture Detection"
    page.window_width = 400
    page.window_height = 300
    
    def go_to_main(e):
        page.views.clear()
        main_menu(page)
    
    page.add(ft.Text("Posture Detection App", size=20, weight=ft.FontWeight.BOLD))
    page.add(ft.ProgressBar(width=300))
    page.update()
    
    time.sleep(2)  # Simulasi loading
    go_to_main(None)

def main_menu(page: ft.Page):
    def start_detection(e):
        page.views.clear()
        posture_detection(page)
    
    page.add(ft.Text("Menu Utama", size=20, weight=ft.FontWeight.BOLD))
    page.add(ft.ElevatedButton("Mulai Deteksi Postur", on_click=start_detection))
    page.update()

def posture_detection(page: ft.Page):
    good_count = 0
    bad_count = 0
    start_time = time.time()

    cap = cv2.VideoCapture(0)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    def process_frame():
        nonlocal good_count, bad_count, start_time
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            keypoints = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            try:
                lm = keypoints.pose_landmarks.landmark
                h, w = image.shape[:2]
                l_shldr_x, l_shldr_y = int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w), int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)
                l_ear_x, l_ear_y = int(lm[mp_pose.PoseLandmark.LEFT_EAR].x * w), int(lm[mp_pose.PoseLandmark.LEFT_EAR].y * h)
                l_hip_x, l_hip_y = int(lm[mp_pose.PoseLandmark.LEFT_HIP].x * w), int(lm[mp_pose.PoseLandmark.LEFT_HIP].y * h)
                neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
                torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)
                if neck_inclination < 40 and torso_inclination < 10:
                    good_count += 1
                else:
                    bad_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 60:
                    print(f"Postur Bagus: {good_count}, Postur Buruk: {bad_count}")
                    good_count, bad_count = 0, 0
                    start_time = time.time()
            except:
                pass
        cap.release()
    
    threading.Thread(target=process_frame, daemon=True).start()
    page.add(ft.Text("Deteksi Postur Sedang Berjalan..."))
    page.update()

def main(page: ft.Page):
    splash_screen(page)

ft.app(target=main)
