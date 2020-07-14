import streamlit as st
import cv2
from PIL import Image,ImageEnhance
import numpy as np 
import os

def detect_faces(our_image):
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = faceCascade.detectMultiScale(gray, 
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return img,faces 

def main():
    """Face Detection App"""
    st.title("Face Detection App")
    activities = ["Detection"]
    choice = st.sidebar.selectbox("Select Activty",activities)

    if choice == 'Detection':
        st.subheader("Face Detection")


        image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
            st.image(our_image)
            st.text("FaceDetection Image")
            result_img,result_faces = detect_faces(our_image)
            st.image(result_img)


if __name__ == '__main__':
    main()