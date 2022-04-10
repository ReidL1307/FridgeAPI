import numpy as np
import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
import time
import json
import pickle
    

hostName = "localhost"
serverPort = 8080

data = json.loads('{"Reid": "false", "Grant": "false", "Luke": "false", "Mom": "false"}')

class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        global data
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(bytes(str(data), "utf-8"))


vc = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("faceTrainer.yml")
labels = {"person_name": 1}
with open("faceLabels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}
    
def face_ai():
    global data
    count = [0, 0, 0, 0]
    pday = 0
    while True:
        day = time.localtime().tm_mday
        ret, frame = vc.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            roiGray = gray[y:y+h, x:x+w]
            roiColor = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 2)
            ids, conf = recognizer.predict(roiGray)
            if conf>=45 and conf <= 85:
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[ids]
                color = (255, 255, 255)
                stroke = 2
                cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
                if name == "grant":
                    count[0] += 1
                    count[1] = 0
                    count[2] = 0
                    count[3] = 0
                    if count[0] > 20:
                        data["Grant"] = "true"
                        count[0] = 0
                elif name == "reid":
                    count[1] += 1
                    count[0] = 0
                    count[2] = 0
                    count[3] = 0
                    if count[1] > 20:
                        data["Reid"] = "true"
                        count[1] = 0
                elif name == "luke":
                    count[2] += 1
                    count[1] = 0
                    count[0] = 0
                    count[3] = 0
                    if count[2] > 20:
                        data["Luke"] = "true"
                        count[2] = 0
                elif name == "mom":
                    count[3] += 1
                    count[1] = 0
                    count[2] = 0
                    count[0] = 0
                    if count[3] > 20:
                        data["Mom"] = "true"
                        count[3] = 0         
        cv2.imshow('farding', frame)
        pday = day
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    webServer = HTTPServer((hostName, serverPort), MyServer)
    x = Thread(target=face_ai)
    y = Thread(target=webServer.serve_forever)
    x.start()
    y.start()
    x.join()
    y.join()

vc.release()
cv2.destroyAllWindows()

