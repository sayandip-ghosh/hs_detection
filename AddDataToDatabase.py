import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate(r'C:\Users\aritr\OneDrive\Desktop\FaceAndGesture_Tracker\serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendanceas-default-rtdb.firebaseio.com/"
})

ref = db.reference('Students')

data = {
    "321654":
        {
            "name": "Anwesha Sen",
            "roll":3755,
            "major": "Machine Learning",
            "starting_year": 2022,
            "total_attendance": 7,
            "standing": "G",
            "year": 2,
            "last_attendance_time": "2022-12-11 00:54:34"
        },
    "852741":
        {
            "name": "Emly Blunt",
            "roll":3795,
            "major": "Economics",
            "starting_year": 2021,
            "total_attendance": 12,
            "standing": "B",
            "year": 3,
            "last_attendance_time": "2022-12-11 00:54:34"
        },
    "963852":
        {
            "name": "Elon Musk",
            "roll":3758,
            "major": "Physics",
            "starting_year": 2020,
            "total_attendance": 7,
            "standing": "G",
            "year": 4,
            "last_attendance_time": "2022-12-11 00:54:34"
        }
}

for key, value in data.items():
    ref.child(key).set(value)