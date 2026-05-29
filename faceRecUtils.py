import cv2
import face_recognition
import math
import numpy as np
import pytz
import sys
import requests
from time import perf_counter
import datetime
from encoding import DEFAULT_FACE_DIR_PATH
from database import parse_all_encodings, session, Student, get_name_from_usn

class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    PROCESS_CURRENT_FRAME: bool = True
    MIN_CONFIDENCE_THRESHOLD: float = 93.5
    last_recognized_usn: list[int] = []

    def __init__(self) -> None:
        pass

    @staticmethod
    def set_allow_attendance(removeTimeConstraints: bool = False):
        if removeTimeConstraints:
            return True
        # Take attendence during a particular interval only
        allow_attendance: bool = True
        attendance_time_start = datetime.time(hour=8)
        attendance_time_end = datetime.time(hour=9, minute=30)

        if not (attendance_time_start <= datetime.datetime.now().time() <= attendance_time_end):
            # Reset Parameters
            allow_attendance = False
            print(f"Attendance taken only during time interval of {attendance_time_start.strftime('%I:%M %p')} to {attendance_time_end.strftime('%I:%M %p')}")
            last_recognized_usn = []

        return allow_attendance

    allow_attendance: bool = set_allow_attendance(removeTimeConstraints=True)

    @staticmethod
    def face_confidence(face_distance, face_match_threshold: float = 0.6) -> float:
        ''' Calculate confidence of recognized face '''
        _range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (_range * 2.0)

        if face_distance > face_match_threshold:
            return round(linear_val * 100, 2)
        else:
            val = (linear_val + ((1.0 - linear_val) * math.pow(((linear_val - 0.5) * 2), 0.2))) * 100
            return round(val, 2)

    @staticmethod
    def update_attendance(usn: int) -> None:
        """ Update attendance of student after a set interval of time """
        try:
            student = session.query(Student).filter(Student.usn == usn).first()
            if student:
                student.attendance += 1
                student.last_attendance_time = datetime.datetime.utcnow()
                session.commit()
                print(f"Attendance updated for {student.name} with usn 1RVU23CSE{student.usn}.")
        except Exception as e:
            print(f"Error updating attendance: {e}")

    @staticmethod
    def annotate_info(frame: cv2.typing.MatLike, usn: int) -> None:
        """ Display information of the recognized person on the frame. """
        student = session.query(Student).filter(Student.usn == usn).first()
        if student and FaceRecognition.allow_attendance:
            font = cv2.FONT_HERSHEY_COMPLEX
            attendance_text = f"Attendance: {student.attendance}"
            cv2.putText(frame, attendance_text, (10, 60), font, 0.8, (0, 255, 0), 2)

            last_attendance_time = student.last_attendance_time
            local_timezone = pytz.timezone('Asia/Kolkata')
            local_time = last_attendance_time.replace(tzinfo=pytz.utc).astimezone(local_timezone)
            last_attendance_time_text = f"Last Attendance Time: {local_time.strftime('%I:%M:%S %p')}"
            cv2.putText(frame, last_attendance_time_text, (10, 90), font, 0.8, (0, 255, 0), 2)

    @staticmethod
    def desired_name_format(usn: int) -> str:
        """ Return upper() of first name only so that it can fix in bbox """
        name: str = get_name_from_usn(usn)
        if name is None:
            return f"Unknown (1RVU23CSE{usn})"
        name = name.split(' ')[0]
        return f"{name.upper()} (1RVU23CSE{usn})"

    def _draw_face_bbox(self, frame: cv2.typing.MatLike) -> None:
        """ Draw a bounding box around the face of the person with their name on it """
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # bring back to original dimensions
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # yellow frame color
            frame_color: tuple = (0, 191, 255)
            font = cv2.FONT_HERSHEY_DUPLEX

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), frame_color, 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), frame_color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

    @staticmethod
    def get_available_cameras():
        """ Get all the indicies of available cameras """
        available_cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()

        if available_cameras:
            print("Available Cameras:", available_cameras)
        else:
            print("No cameras found.")
        return available_cameras

    @staticmethod
    def phone_camera_feed(_ip_webcam_url: str = 'https://100.77.184.63:8080') -> cv2.typing.MatLike:
        _ip_webcam_url += '/shot.jpg'
        res = requests.get(_ip_webcam_url)
        imgArr = np.array(bytearray(res.content), dtype=np.uint8)
        img = cv2.imdecode(imgArr, -1)
        return img

    def run_recognition(self) -> cv2.typing.MatLike:
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit("Video Source Not Found ...")

        # Initialize FPS variables
        vidFps = 0
        prev_time = perf_counter()
        frames = 0

        # Load all the known encodings and known names from database into a list for comparison
        self.known_face_names, self.known_face_encodings = parse_all_encodings()

        while True:
            ret, frame = video_capture.read()
            # frame = FaceRecognition.phone_camera_feed()                    # Get footage from external device
            frame = cv2.flip(frame, 1)

            # Calculate FPS
            current_time = perf_counter()
            frames += 1
            if current_time - prev_time >= 1:
                vidFps = frames / (current_time - prev_time)
                prev_time = current_time
                frames = 0

            if self.PROCESS_CURRENT_FRAME:
                # resize the frame to 1/4 size to save computation power
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Find face locations and encodings in the current frame
                self.face_locations = face_recognition.face_locations(rgb_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_frame, self.face_locations)

                self.face_names = []

                usn: int = -1
                confidence: float = 0.0

                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)

                    # lower the face distance, better the match
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    if len(face_distances) > 0: 
                        # get index of lowest face distance
                        best_match_index = np.argmin(face_distances)
                        
                        if matches[best_match_index]:
                            usn = self.known_face_names[best_match_index]
                            confidence: float = self.face_confidence(face_distances[best_match_index])

                            # Update Attendance With Set Flags
                            if FaceRecognition.allow_attendance and (usn not in self.last_recognized_usn) and (confidence > self.MIN_CONFIDENCE_THRESHOLD):
                                self.update_attendance(usn)
                                self.last_recognized_usn.append(usn)
                                # draw frame if confidence > THRESHOLD
                                self._draw_face_bbox(frame)
                            
                    else: 
                        continue

                    self.face_names.append(self.desired_name_format(usn))

            # Process alternate frames
            self.PROCESS_CURRENT_FRAME = not self.PROCESS_CURRENT_FRAME

            self._draw_face_bbox(frame)

            # Draw attendance information for person in the current frame
            FaceRecognition.annotate_info(frame, usn)

            # Annotate FPS
            cv2.putText(frame, "FPS: {:.2f}".format(vidFps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            try:
                cv2.imshow("FACE RECOGNITION", frame)
            except cv2.error as e:
                print("Encountered cv2.error ")
                continue
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    pass