#import kivy dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

#import kivy UI/UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

#other kivy dependencies
from  kivy.clock import Clock #get real time 
from kivy.graphics.texture import Texture
from kivy.logger import Logger

#import other dependencies
import cv2
import os
import tensorflow as tf
from layers import L1Dist
import numpy as np

#build app layout
class CamApp(App):

    def build(self):
        #main layout components
        self.webCam = Image(size_hint=(1,.8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1,.1))
        self.verification_label = Label(text = "Verification Uninitiated", size_hint=(1,.1))

        #add items to the layout
        layout = BoxLayout(orientation='vertical')

        #layout is in sequential order
        layout.add_widget(self.webCam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        #load tensorflow/keras model
        self.model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist': L1Dist})

        #setup video capture device
        self.capture = cv2.VideoCapture(0)

        #real time feed
        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout

    #run continuously to get the webcam feed
    def update(self, *args):
        # read your frame from opencv
        #same as in jupyter notebook

        ret, frame = self.capture.read()

        #cut frame size to 250x250 pixels
        frame = frame[50:50+250, 230:230+250, :]

        #flip horizontal and convert image to texture

        #converting our raw opencv image array to a texture for rendering and then setting our image equal to thatt texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf,  colorfmt='bgr', bufferfmt='ubyte')
        self.webCam.texture = img_texture

    #copy the preprocess function from jupyter notebook
    
    #we need this because dataset needs to preprocessed before passing to tensorflow

    def preprocess(self, file_path):

        #read our image from file path
        byte_img = tf.io.read_file(file_path)
        
        #load the image using tensorflow decode
        img = tf.io.decode_jpeg(byte_img)
        
        #resizing image to be 100x100x3 
        img = tf.image.resize(img, (100,100))
        
        #scale image to be between 0 and 1
        img = img/255.0
        return img

    #bring the verify function from jupyter 

    # to verify a person
    def verify(self, *args):
        
        # what the limit is before out prediction is considered positive
        detection_threshold = 0.99

        # what proportions  of predictions need to be positive for a match
        verification_threshold = 0.9

        #capture input image from our webcam
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')

        #save in the filepath
        ret, frame = self.capture.read()
        frame = frame[50:50+250, 230:230+250, :]
        cv2.imwrite(SAVE_PATH, frame)


        #build results array
        results=[]
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img=self.preprocess(os.path.join('application_data', 'input_image','input_image.jpg'))
            validation_img=self.preprocess(os.path.join('application_data','verification_images',image))
            
            #make predictions
            result=self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)
            
        #detection threshold: metric above 50% is considered positive
        detection = np.sum(np.array(results) > detection_threshold)
        
        #verification threshold: proportion of positive predictions/total positive samples
        
        verification = detection/len(os.listdir(os.path.join('application_data', 'verification_images')))
        verified = verification > verification_threshold
        
        # set the verification text to verified or unverified
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'

        # Log out details
        Logger.info(results)
        Logger.info(np.sum(np.array(results) > 0.5))
        Logger.info(np.sum(np.array(results) > 0.4))
        Logger.info(np.sum(np.array(results) > 0.8))
        Logger.info(np.sum(np.array(results) > 0.2))

        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)



        return results, verified


if __name__ == '__main__':
    CamApp().run()

