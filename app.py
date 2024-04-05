from flask import Flask, render_template, Response
import pickle
import face_recognition
import cv2
import mediapipe as mp
import numpy as np
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime

app = Flask(__name__)

# sign
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

camera = cv2.VideoCapture(0)
camera.set(3, 640)
camera.set(4, 480)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'Thumbs Up', 1: 'Thumbs Down', 2: 'Closed Fist',
               3: 'Open Palm', 4: 'Nice', 5: 'Swag', 6: 'Victory', 7: 'Point Up'}


# sign
def open_Sign():
    while True:
        data_aux = []
        x_ = []
        y_ = []
        isTrue, frame = camera.read()
        if not isTrue:
            break
        else:
            H, W, _ = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,  # image to draw
                        hand_landmarks,  # model output
                        mp_hands.HAND_CONNECTIONS,  # hand connections
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                prediction = model.predict([np.asarray(data_aux)])

                predicted_character = labels_dict[int(prediction[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera.release()


# atd
studentInfo = None
studentInfo_set = False


def gen_frame():
    global studentInfo
    global datetimeObject
    global imgStudent
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            bucket = storage.bucket()
            print("Loading Encode File ...")
            file = open('EncodeFile.p', 'rb')
            encodeListKnownWithIds = pickle.load(file)
            file.close()
            encodeListKnown, studentIds = encodeListKnownWithIds
            print("Encode File Loaded")
            counter = 0
            id = -1
            imgStudent = []

            imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            faceCurFrame = face_recognition.face_locations(imgS)
            encodeCurFrame = face_recognition.face_encodings(
                imgS, faceCurFrame)
            if faceCurFrame:
                for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                    matches = face_recognition.compare_faces(
                        encodeListKnown, encodeFace)
                    faceDis = face_recognition.face_distance(
                        encodeListKnown, encodeFace)

                    matchIndex = np.argmin(faceDis)

                    if matches[matchIndex]:

                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                        # frame = cvzone.cornerRect(frame, bbox, rt=1)
                        id = studentIds[matchIndex]
                        if counter == 0:
                            counter = 1
                            studentInfo_set = True

                if counter != 0:

                    if counter == 1:
                        # Get the Data
                        studentInfo = db.reference(f'Students/{id}').get()
                        print(studentInfo)
                        # Get the Image from the storage
                        blob = bucket.get_blob(f'Images/{id}.png')
                        array = np.frombuffer(
                            blob.download_as_string(), np.uint8)
                        imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
                        # Update data of attendance
                        datetimeObject = datetime.strptime(studentInfo['last_attendance_time'],
                                                           "%Y-%m-%d %H:%M:%S")
                        secondsElapsed = (
                            datetime.now() - datetimeObject).total_seconds()
                        print(datetimeObject)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


cred = credentials.Certificate(
    r'serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendanceas-default-rtdb.firebaseio.com/",
    'storageBucket': "faceattendanceas.appspot.com"
})


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/video2')
def video2():
    return Response(open_Sign(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/sign')
def btn():
    return render_template('sign.html')

# atd


@app.route('/atd')
def atd():
    return render_template('atd.html')


@app.route('/start')
def start():
    return render_template('new.html')


@app.route('/video')
def video():
    return Response(gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop')
def stop():
    global studentInfo
    global datetimeObject
    global imgStudent
    global studentInfo_set
    if studentInfo_set == False:
        return render_template('atd.html')

    name = studentInfo['name']
    time = datetimeObject
    roll = studentInfo['roll']
    return render_template('atd.html', name=name, time=time, roll=roll)


if __name__ == "__main__":
    app.run(debug=True)
