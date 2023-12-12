from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.uix.screenmanager import ScreenManager, Screen, FallOutTransition
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.button import Button
import time
from kivy.uix.image import Image
import os
from PIL import Image as PilImage
import cv2
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class WindowManager(ScreenManager):
    pass
class MainWindow(Screen):
    pass 
class SecondWindow(Screen):
    transition = FallOutTransition()

class ThirdWindow(Screen):
    def capture_image(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        IMAGES_FOLDER = 'images'
        camera = self.ids['camera']
        timestr = time.strftime("%Y%m%d_%H%M%S")
        #check if the folder is existed
        if not os.path.exists(IMAGES_FOLDER):
            os.makedirs(IMAGES_FOLDER)

        # Set the full path for the captured image
        image_path = os.path.join(IMAGES_FOLDER, "IMG_{}.png".format(timestr))
        camera.export_to_png(image_path)
        print("Captured")
        # Resize the image to 1000x1000 pixels
        resized_image = PilImage.open(image_path)
        resized_image = resized_image.resize((1000, 1000))
        resized_image.save(image_path)
        # Set the image source in FourthWindow
        fourth_screen = self.manager.get_screen("Fourth")
        fourth_screen.set_captured_image(image_path)


model = tf.keras.models.load_model('good.h5')

class FourthWindow(Screen):
    image_path = 'C:\my_data\myenv_file\Thesis_app\images'
    

    def set_captured_image(self, image_path):
        self.captured_image_path = image_path
        self.ids['captured_image'].source = image_path


    def predict_not_cooked_or_cooked(self):
        image_path = self.captured_image_path
        threshold_value = 0.5

        img = cv2.imread(image_path)
        img = cv2.resize(img, (244, 244))
        img = img.astype(np.float32) / 255.0
        img_array = np.expand_dims(img, axis=0)

        prediction = model.predict(img_array)

        label = 'COOKED' if prediction[0][0] > threshold_value else 'NOT COOKED'
        confidence = prediction[0][0]
        percentage = confidence * 100

        return [(label, percentage)]

    def predict_cooked_probability(self):

        # Call the new prediction function
        predictions = self.predict_not_cooked_or_cooked()

        # Update the prediction label in FifthWindow
        predictions = self.predict_not_cooked_or_cooked()
        if predictions:
            prediction_label, _ = predictions[0]
            fifth_screen = self.manager.get_screen("Fifth")
            fifth_screen.set_captured_image(self.captured_image_path)
            fifth_screen.set_prediction_label(prediction_label)


class FifthWindow(Screen):
    def __init__(self, **kwargs):
        super(FifthWindow, self).__init__(**kwargs)
        self.image_path = 'C:\my_data\myenv_file\Thesis_app\images'
        self.prediction_label_text = "Value"  # Initialize the attribute
    
    def set_captured_image(self, image_path):
        self.captured_image_path = image_path
        self.ids['captured_image'].source = image_path

    def set_prediction_label(self, prediction_label):
            self.prediction_label_text = f'{prediction_label}'
            print(f"Setting Prediction Label: {self.prediction_label_text}")
            self.ids['prediction_label'].text = self.prediction_label_text

class MyApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Green"

        return Builder.load_file("app.kv")

if __name__ == '__main__':
    MyApp().run()
