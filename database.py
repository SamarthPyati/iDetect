from pathlib import Path
import os
import numpy as np
import pickle
import datetime
from PIL import Image
from io import BytesIO
import numpy as np
from sqlalchemy import (
    create_engine, 
    Column, 
    String, 
    Integer, 
    CHAR, 
    LargeBinary, 
    PickleType, 
    Enum,
    DateTime
)
from sqlalchemy.orm import sessionmaker, declarative_base
from encoding import encode_face, get_face_encodings, DEFAULT_FACE_DIR_PATH
Base = declarative_base()

class Student(Base):
    __tablename__ = "students"
    
    GENDERS = ("Male", "Female", "Others")

    usn = Column("usn", Integer, primary_key=True, autoincrement=True, index=True)
    name = Column("name", String, nullable=False)
    course = Column("course", String, nullable=False)
    year_join = Column("year of join", Integer, nullable=False)
    attendance = Column("attendance", Integer, nullable=False, default=0)
    section = Column("section", CHAR, nullable=False)
    gender = Column("gender", Enum(*GENDERS), nullable=False)
    face_image = Column("face image", LargeBinary, nullable=False, unique=True)
    face_encoding = Column("face encodings", PickleType, nullable=False, unique=True)
    last_attendance_time = Column("last attendance time", DateTime, nullable=False, default=datetime.datetime.utcnow())

    def __repr__(self) -> str:
        return f"{self.usn}, {self.name}, {self.course} {self.year_join}, Section {self.section}"


engine = create_engine("sqlite:///database.db", echo=False)
Base.metadata.create_all(bind=engine)
Session = sessionmaker(bind=engine)
session = Session()

def get_face_image(face_image_name: Path | str, face_path: Path | str = DEFAULT_FACE_DIR_PATH) -> bytes:
    """ Return images as bytes """
    try:
        face_path = os.path.join(DEFAULT_FACE_DIR_PATH, face_image_name)
        if not os.path.exists(face_path):
            raise FileNotFoundError("Image file not found")
        else:
            with open(face_path, "rb") as f:
                image = f.read()
            return image
    except Exception as e:
        print(f"Fetch image error: {e}")


def create_student(usn: int, name: str, course: str, year_join: int, attendance: int, section: str, gender: str, image_name: str) -> None:
    try:
        # get the face image
        face_image = get_face_image(image_name)
        face_encodings = get_face_encodings(image_name)

        if face_image is None:
            print(f"Image file '{image_name}' not found. Skipping student creation.")
            return 
            
        # Check if the student already exists in the database
        existing_student = session.query(Student).filter(Student.usn == usn).first()
        if existing_student:
            print(f"Student with name {usn} already exists in the database.")
            return

        # Create a new student object
        student = Student(
            usn=usn,
            name=name,
            course=course,
            year_join=year_join,
            attendance=attendance,
            section=section,
            gender=gender,
            face_image=face_image,
            face_encoding=face_encodings
        )

        # Add the student to the session and commit the transaction
        with session:
            session.add(student)
            session.commit()
            print(f"Student {name} added with USN {usn}.")
    except FileNotFoundError as e:
        print(f"Fetch image error: {e}")
    except Exception as e:
        print(f"Error creating student: {e}")

def get_all_student() -> None:
    try:
        with session: 
            result = session.query(Student).all()
            if result:
                for r in result:
                    print(r)
            else:
                print("No data")
    except Exception as e:
        print(f"Error fetching all students: {e}")

def view_face(usn: int) -> None:
    try:
        student = session.query(Student).filter(Student.usn == usn).first()    
        if student:
            student_image = Image.open(BytesIO(student.face_image))
            student_image.show()
        else:
            print("Student not found.")
    except FileNotFoundError as e:
        print(f"Image file not found: {e}")
    except Exception as e:
        print(f"Error viewing face: {e}")

def parse_encoding(usn: int) -> tuple[str, np.ndarray]:
    try:
        student = session.query(Student).filter(Student.usn == usn).first()
        if student and student.face_encoding:
            return student.face_encoding
        else:
            print(f"No face encoding found for student with USN: {usn}")
    except Exception as e:
        print(f"Error parsing encodings: {e}")

def get_name_from_usn(usn: int) -> str | None:
    try:
        # if -1 then Unknown
        if usn == -1:
            return "Unknown"

        student = session.query(Student).filter(Student.usn == usn).first()
        if student:
            return student.name
        else:
            return None
    except Exception as e:
        print(f"Error finding name: {e}")

def parse_all_encodings() -> tuple[list[str], list[np.ndarray]]:
    """ parse all encodings and returns tuple of (image_names, face_encodings) """
    try:
        known_face_names: list[str] = []
        known_face_encodings = []
        students = session.query(Student).all()
        for student in students:
            if student.face_encoding:
                im_name, enc = student.face_encoding
                im_name, _ = os.path.splitext(im_name)
                known_face_names.append(im_name)
                known_face_encodings.append(enc)
        return known_face_names, known_face_encodings
    except Exception as e:
        print(f"Error parsing all encodings: {e}")

if __name__ == "__main__":
    # Testing 
    try:
        # # adding all the people in database
        # create_student(400, "Samarth Sanjay Pyati", "B.Tech CSE", 2023, 0, "F", "Male", "400.jpg")
        # create_student(87, "Atharv Bhujannavar", "B.Tech CSE", 2023, 0, "I", "Male", "087.jpeg")
        # create_student(426, "Shashwath Jain H.P", "B.Tech CSE", 2023, 0, "F", "Male", "426.jpeg")
        # create_student(418, "Sharan S Gowda", "B.Tech CSE", 2023, 0, "I", "Male", "418.jpeg")
        # create_student(490, "Sushruth", "B.Tech CSE", 2023, 0, "I", "Male", "490.jpeg")
        # create_student(48, "Anagha", "B.Tech CSE", 2023, 0, "H", "Female", "048.jpeg")
        get_all_student()
    except Exception as e:
        print(f"Error occurred during testing: {e}")
