# ต้องติดตั้ง cv2 แบบใช้ gpu วิธีติดตั้ง https://www.youtube.com/watch?v=HsuKxjQhFU0
# pip install cupy 
# pip insall tensorflow 


import cv2
import cupy as cp
import numpy as np
from keras.models import load_model
import tensorflow as tf
import time

class FaceRecognizer:
    def __init__(self, camera_index=0):
        # keras_Model.h5 and labels.txt ให้ไปtrain model จาก https://teachablemachine.withgoogle.com/
        self.model = load_model("model/keras_Model.h5", compile=False)
        self.class_names = open("model/labels.txt", "r").readlines()
        
        # haarcascade_frontalface_default_cuda.xml โหลดจาก https://github.com/opencv/opencv/blob/4.x/data/haarcascades_cuda/haarcascade_frontalface_default.xml
        
        self.face_cascade = cv2.cuda.CascadeClassifier_create("haarcascade_frontalface_default.xml")
        self.camera = cv2.VideoCapture(camera_index)
        self.confidence_threshold = 0.5  # ค่าเปรียบเทียมกับคำตอบ
        self.start_time = time.time()
        self.frame_counter = 0 

    def preprocess_image(self, image):
        # ประมวลผลภาพล่วงหน้าโดยการปรับขนาดและทำให้เป็นมาตรฐาน  ภาษาอังกฤษ Preprocess image by resizing and normalizing
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        images = cp.asarray(image, dtype=cp.float32).reshape(1, 224, 224, 3)
        images = (images / 127.5) - 1
        return images
    
    def recognize_faces(self):
        while True:
            ret, image = self.camera.read()
            if not ret:
                print("ไม่สามารถจับภาพเฟรมได้ , Failed to capture frame")
                break
            
            self.frame_counter += 1
            # กลับด้าน image
            flipped_image = cv2.flip(image, 1)
            gray = cv2.cvtColor(flipped_image, cv2.COLOR_BGR2GRAY)
            
            # แปลงเป็น GPU Mat
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(gray)
            
            # ทำการตรวจจับใบหน้าบน GPU
            faces = self.face_cascade.detectMultiScale(gpu_image)
            
            #ดาวน์โหลดผลลัพธ์จาก GPU ไปยัง CPU
            faces = faces.download()
            
            images = self.preprocess_image(flipped_image)
            prediction = self.model.predict(cp.asnumpy(images), verbose=False) # predict 
            index = np.argmax(prediction)  
            confidence_score = prediction[0][index]

            #  
            
            if confidence_score < self.confidence_threshold:
                class_name = "Unknown"
            else:
                class_name = self.class_names[index][2:]

            x, y, w, h = faces[0][0]  # แยกพิกัดของใบหน้าแรกที่ตรวจพบ
            cv2.rectangle(flipped_image, (x, y), (x+w, y+h), (255, 0, 0), 2) # ตีกรอบบนใบหน้า
            text = f"{class_name} Score: {int(confidence_score * 100)}%"
            cv2.putText(flipped_image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)# แสดงผลของการตรวจผม
            
            # คำนวณ FPS
            elapsed_time = time.time() - self.start_time
            fps = self.frame_counter / elapsed_time
            cv2.putText(flipped_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # แสดงWebcam
            cv2.imshow("Webcam Image", flipped_image)
            
            keyboard_input = cv2.waitKey(1)
            if keyboard_input == 27:
                break

        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # set gpu for tensor flow
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    # เรียกใช้ class และเริ่มprogram
    recognizer = FaceRecognizer()
    recognizer.recognize_faces()
