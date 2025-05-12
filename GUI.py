import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import glob
from joblib import load
from openpyxl import Workbook
import cv2
import numpy as np
from Others.SVM import *

#load trained model
clf = load('Others/svm_model.joblib')

#GUI class
class PredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Human vs Non-Human Classifier")
        self.image_paths = []
        self.index = 0
        self.predictions = []

        self.label = tk.Label(root, text="Choose a folder with 10 positive and 10 negative images")
        self.label.pack()

        #add load directory button
        self.choose_btn = tk.Button(root, text="Load Directory", command=self.load_directory)
        self.choose_btn.pack()

        #add canvas for showing images
        self.canvas = tk.Label(root)
        self.canvas.pack()

        self.prediction_label = tk.Label(root, text="", font=("Helvetica", 14))
        self.prediction_label.pack()

        #load next image
        self.next_btn = tk.Button(root, text="Next", command=self.next_image)
        self.next_btn.pack()

    #from the directory load one image
    def load_directory(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        self.image_paths = sorted(glob.glob(os.path.join(folder, "*.*")))
        self.index = 0
        self.predictions.clear()
        self.display_image()

    #show the image in the canvas
    def display_image(self):
        #if it is the last image, save the result to .xlsx
        if self.index >= len(self.image_paths):
            self.save_predictions()
            self.prediction_label.config(text="Done! Predictions saved.")
            return

        #show image in canvas
        path = self.image_paths[self.index]
        img = Image.open(path)
        img = img.resize((256, 512))  # scale up for better view
        tk_img = ImageTk.PhotoImage(img)
        self.canvas.configure(image=tk_img)
        self.canvas.image = tk_img

        #execute the prediction with trained svm
        pred = predict_with_svm(clf, path)
        label = "Human" if pred == 1 else "Non-Human"
        self.prediction_label.config(text=f"{os.path.basename(path)} → Prediction: {label}")

        #save the prediction result to array
        self.predictions.append((os.path.basename(path), pred))

    #show the next image
    def next_image(self):
        self.index += 1
        self.display_image()

    #save the prediction result to .xlsx file
    def save_predictions(self):
        wb = Workbook()
        ws = wb.active
        ws.title = "Predictions"
        ws.append(["Filename", "Prediction (1=Human, 0=Non-Human)"])
        for filename, pred in self.predictions:
            ws.append([filename, pred])
        wb.save("predictions.xlsx")

#run the GUI 
if __name__ == "__main__":
    root = tk.Tk()
    gui = PredictionGUI(root)
    root.mainloop()
