from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import matplotlib.pyplot as plt


class Face2D:

    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.landmarks = {}
        self.face_colors = {}
        self.face_colors['face'] = (235, 64, 52)
        self.face_colors['right_eyebrow'] = (245, 241, 39)
        self.face_colors['left_eyebrow'] = (54, 255, 36)
        self.face_colors['nose'] = (25, 255, 236)
        self.face_colors['right_eye'] = (20, 67, 252)
        self.face_colors['left_eye'] = (132, 3, 252)
        self.face_colors['mouth'] = (252, 3, 186)

    def get_landmarks(self, image_path):
        """[gets the coordinates of 7 landmark features when given the path to a face image]

        Arguments:
            image_path {string} -- [path to image]

        Returns:
            [dictionary {string -> np array of coordinates}] -- [mapping of landmarks to positions]
        """

        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(
            "shape_predictor_68_face_landmarks.dat")

        # load the input image, resize it, and convert it to grayscale
        self.image = imutils.resize(self.image, width=500)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale image
        rects = detector(gray, 1)

        i, rect = rects[0]  # take the first face it finds
        shape = predictor(gray, rect)
        # gets the coordinates as a numpy array
        shape = face_utils.shape_to_np(shape)
        # separate into facial features
        face_parts = {}
        face_parts['face'] = shape[0:17]
        face_parts['right_eyebrow'] = shape[17:22]
        face_parts['left_eyebrow'] = shape[22:27]
        face_parts['nose'] = shape[27:36]
        face_parts['right_eye'] = shape[36:42]
        face_parts['left_eye'] = shape[42:48]
        face_parts['mouth'] = shape[48:]

        return face_parts

    def create_face_2D(self, landmarks):
        '''
            take in landmarks, add additional nodes, output landmarks 
        '''

    def plot_face(self, landmarks):
        ''' plots the face in a normalized coordinate setting, centered, for testing '''

    def draw_landmarks(self):
        """[draws landmarks on face image]

        Arguments:
            face_image {cv2 image} -- [image of face to draw on]
            landmarks {dictionary {string -> np array of coordinates}} -- [dict of face landmarks]
        """

        for part, coords in self.landmarks.items():
            for (x, y) in coords:
                cv2.circle(self.image, (x, y), 1, self.face_colors[part], -1)
                print(str(x) + ", " + str(x))
        # show the output image with the face detections + facial landmarks
        cv2.imshow("Output", self.image)
        cv2.waitKey(0)


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    # ap.add_argument("-p", "--shape-predictor", required=True,
    #                 help="path to facial landmark predictor")
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    args = vars(ap.parse_args())
