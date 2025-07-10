import cv2
import time
import math as m
import mediapipe as mp
import logging
import pandas as pd
from datetime import datetime

# Inisialisasi Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Fungsi jarak
def findDistance(x1, y1, x2, y2):
    return m.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Fungsi sudut
def findAngle(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    return m.degrees(m.atan2(dy, dx))

# Fungsi peringatan
def sendWarning():
    print("Peringatan: Postur buruk terlalu lama!")
    # Tambahkan suara beep jika diperlukan
    # winsound.Beep(1000, 500)

# Logging
logging.basicConfig(level=logging.INFO)

# Warna
colors = {
    "blue": (255, 127, 0),
    "red": (50, 50, 255),
    "green": (127, 255, 0),
    "yellow": (0, 255, 255)
}

# Menu utama
def main_menu():
    print("\n=== Menu Utama ===")
    print("1. Memulai Deteksi")
    print("2. Pengaturan")
    print("3. Keluar")
    choice = input("Pilih menu (1/2/3): ")
    return choice

# Simpan data ke Excel
def save_to_excel(data, filename="posture_data.xlsx"):
    try:
        # Coba baca file Excel jika sudah ada
        df = pd.read_excel(filename)
    except FileNotFoundError:
        # Jika file tidak ada, buat DataFrame kosong
        df = pd.DataFrame(columns=["Nama", "Umur", "Aksi", "Waktu Pengukuran", 
                                   "Durasi Pengukuran (detik)", 
                                   "Jumlah Postur Baik", "Jumlah Postur Buruk", 
                                   "Lama Postur Baik (detik)", "Lama Postur Buruk (detik)"])
    
    # Tambahkan data baru ke DataFrame
    new_row = pd.DataFrame([data])
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Simpan ke file Excel
    df.to_excel(filename, index=False)
    print(f"Data berhasil disimpan ke {filename}")

# Memulai deteksi
def start_detection():
    print("\n=== Memulai Deteksi ===")
    name = input("Masukkan nama: ")
    age = input("Masukkan umur: ")
    action = input("Apa aksi yang sedang dilakukan? (Contoh: Duduk, Berdiri): ")
    duration = int(input("Berapa lama pengukuran (dalam detik)? "))

    print(f"\nMemulai deteksi untuk {name}, Umur: {age}, Aksi: {action}, Durasi: {duration} detik.")
    time.sleep(1)  # Jeda sebelum memulai deteksi

    # Inisialisasi variabel
    good_frames = 0
    bad_frames = 0
    neck_max = 40
    torso_max = 10
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Webcam tidak dapat dibuka")
        return

    start_time = time.time()
    end_time = start_time + duration

    try:
        while time.time() < end_time and cap.isOpened():
            success, image = cap.read()
            if not success:
                logging.warning("Frame kosong")
                break

            h, w = image.shape[:2]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            keypoints = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                lm = keypoints.pose_landmarks
                lmPose = mp_pose.PoseLandmark

                # Koordinat landmark
                l_shldr = (int(lm.landmark[lmPose.LEFT_SHOULDER].x * w), int(lm.landmark[lmPose.LEFT_SHOULDER].y * h))
                r_shldr = (int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w), int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h))
                l_ear = (int(lm.landmark[lmPose.LEFT_EAR].x * w), int(lm.landmark[lmPose.LEFT_EAR].y * h))
                l_hip = (int(lm.landmark[lmPose.LEFT_HIP].x * w), int(lm.landmark[lmPose.LEFT_HIP].y * h))

                # Hitung offset
                offset = findDistance(*l_shldr, *r_shldr)
                if offset < 100:
                    cv2.putText(image, f"{int(offset)} Samping", (w - 150, 30), font, 0.9, colors["green"], 2)
                else:
                    cv2.putText(image, f"{int(offset)} Depan/Belakang", (w - 150, 30), font, 0.9, colors["red"], 2)

                # Hitung sudut
                neck_inclination = findAngle(*l_shldr, *l_ear)
                torso_inclination = findAngle(*l_hip, *l_shldr)

                # Evaluasi postur
                if neck_inclination < neck_max and torso_inclination < torso_max:
                    good_frames += 1
                    color = colors["green"]
                else:
                    bad_frames += 1
                    color = colors["red"]

                # Gambar landmark
                points = [l_shldr, l_ear, l_hip]
                for point in points:
                    cv2.circle(image, point, 7, colors["yellow"], -1)

                # Tampilkan teks
                angle_text = f'Leher : {int(neck_inclination)}  Batang Tubuh : {int(torso_inclination)}'
                cv2.putText(image, angle_text, (10, 30), font, 0.9, color, 2)

                # Tampilkan hasil sementara
                total_time = time.time() - start_time
                good_time = (good_frames / 30) if total_time > 0 else 0
                bad_time = (bad_frames / 30) if total_time > 0 else 0

                cv2.putText(image, f"Postur Bagus: {round(good_time, 1)}s", (10, h - 60), font, 0.9, colors["green"], 2)
                cv2.putText(image, f"Postur Buruk: {round(bad_time, 1)}s", (10, h - 30), font, 0.9, colors["red"], 2)

                if bad_time > 180:
                    sendWarning()

            except AttributeError as e:
                cv2.putText(image, "Tidak Ada Orang", (10, h - 20), font, 0.9, colors["red"], 2)
                logging.error(f"Error: {e}")

            cv2.imshow('Membaca Postur Tubuh', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        # Hasil akhir
        print("\n=== Hasil Pengukuran ===")
        print(f"Nama: {name}")
        print(f"Umur: {age}")
        print(f"Aksi: {action}")
        print(f"Lama Pengukuran: {duration} detik")
        print(f"Jumlah Postur Baik: {good_frames}")
        print(f"Jumlah Postur Buruk: {bad_frames}")
        print(f"Lama Postur Baik: {round(good_time, 1)} detik")
        print(f"Lama Postur Buruk: {round(bad_time, 1)} detik")

        # Simpan hasil ke Excel
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = {
            "Nama": name,
            "Umur": age,
            "Aksi": action,
            "Waktu Pengukuran": timestamp,
            "Durasi Pengukuran (detik)": duration,
            "Jumlah Postur Baik": good_frames,
            "Jumlah Postur Buruk": bad_frames,
            "Lama Postur Baik (detik)": round(good_time, 1),
            "Lama Postur Buruk (detik)": round(bad_time, 1)
        }
        save_to_excel(data)

    finally:
        cap.release()
        cv2.destroyAllWindows()

# Pengaturan
def settings():
    print("\n=== Pengaturan ===")
    print("Fitur pengaturan belum tersedia.")
    input("Tekan Enter untuk kembali ke menu utama...")

# Program utama
def main():
    while True:
        choice = main_menu()
        if choice == "1":
            start_detection()
        elif choice == "2":
            settings()
        elif choice == "3":
            print("\nKeluar dari program. Sampai jumpa!")
            break
        else:
            print("Pilihan tidak valid. Silakan coba lagi.")

if __name__ == "__main__":
    main()