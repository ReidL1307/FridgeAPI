import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
imageDir = os.path.join(BASE_DIR, "images")

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

currentId = 0
labelIds = {}
yLabels = []
xTrain = []

for root, dirs, files in os.walk(imageDir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			label = os.path.basename(root).replace(" ", "-").lower()
			if not label in labelIds:
				labelIds[label] = currentId
				currentId += 1
			ids = labelIds[label]
			pil_image = Image.open(path).convert("L")
			size = (550, 550)
			finalImage = pil_image.resize(size, Image.Resampling.LANCZOS)
			imageArray = np.array(finalImage, "uint8")
			print(imageArray)
			faces = faceCascade.detectMultiScale(imageArray, 1.1, 4)

			for (x,y,w,h) in faces:
				roi = imageArray[y:y+h, x:x+w]
				xTrain.append(roi)
				yLabels.append(ids)




with open("faceLabels.pickle", 'wb') as f:
	pickle.dump(labelIds, f)

recognizer.train(xTrain, np.array(yLabels))
recognizer.save("faceTrainer.yml")
