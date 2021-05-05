# Importing OpenCV and Numpy for video Capture, recognition and arrays
import cv2
import numpy as np

# Using keras to import the .json file from training model to detect
from keras.models import model_from_json
from keras.preprocessing import image

# importing trained dataset
dataset = model_from_json(open("training.json", "r").read())
dataset.load_weights('training.h5')

# Using Haar Cascade to detect and recognize face from real-time video capture
haar_cascade_frontalface = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Video Capture
capture = cv2.VideoCapture(0)

# While loop for image conversion
while True:
    rturn, img=capture.read()
    if not rturn:
        continue
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # For loop for creating bounding box for the face
    detected_faces = haar_cascade_frontalface.detectMultiScale(gray_image, 1.32, 5)
    for (x, y, w, h) in detected_faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), thickness=7)
        roi_gray = gray_image[y:y+w, x:x+h] # Extracting region of interest i.e Face from the video capture in real-time
        roi_gray = cv2.resize(roi_gray, (48, 48)) # Resizing
        image_pixels = image.img_to_array(roi_gray) # Conversion of image to array
        image_pixels = np.expand_dims(image_pixels, axis=0)
        image_pixels /= 255

        # Creating Prediction model for CNN to recognise
        predict_model = dataset.predict(image_pixels)

        max_index = np.argmax(predict_model[0])

        # Classifying Emotions into 7 for differentiation and to display into the capture
        emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
        predict_emotions = emotions[max_index]

        # Using putText function to display emotions on display
        cv2.putText(img, predict_emotions, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Resizing Capture Video to 720 Pixels
    image_resize = cv2.resize(img, (1280, 720))
    cv2.imshow('Emotion Detectioon', image_resize)

    # Sending Keystroke "Q" for Quitting the Application
    if cv2.waitKey(10) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()