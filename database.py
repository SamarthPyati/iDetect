from pathlib import Path
import os
import numpy as np
import pickle
import datetime
from PIL import Image, ImageTk
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
    year_join = Column("year of joining", Integer, nullable=False)
    attendance = Column("attendance", Integer, nullable=False, default=0)
    section = Column("section", CHAR, nullable=False)
    gender = Column("gender", Enum(*GENDERS), nullable=False)
    face_image = Column("face image", LargeBinary, nullable=False, unique=True)
    face_encoding = Column("face encodings", PickleType, nullable=False, unique=True)
    last_attendance_time = Column("last attendance time", DateTime, nullable=False, default=datetime.datetime.utcnow())

    def __repr__(self) -> str:
        return f"{self.usn:03d}, {self.name}, {self.course} {self.year_join}, Section {self.section}"


engine = create_engine("sqlite:///database.db", echo=False)
Base.metadata.create_all(bind=engine)
Session = sessionmaker(bind=engine)
session = Session()


def get_face_image(face_image_name: str, file_path: Path | str = DEFAULT_FACE_DIR_PATH) -> bytes:
    """ Return images as bytes """
    try:
        file_path = os.path.join(DEFAULT_FACE_DIR_PATH, face_image_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError("Image file not found")
        else:
            with open(file_path, "rb") as f:
                image = f.read()
            return image
    except Exception as e:
        print(f"Fetch image error: {e}")


def create_student(usn: int, name: str, course: str, year_join: int, section: str, gender: str, fp: Path | str) -> None:
    try:
        path, im_name = os.path.split(fp)

        face_image = get_face_image(im_name, path)
        face_encodings = get_face_encodings(im_name, path)

        if face_image is None:
            print(f"Image file '{name}' not found. Skipping student creation.")
            return

        # Check if the student already exists in the database
        existing_student = session.query(Student).filter(Student.usn == usn).first()
        if existing_student:
            print(f"Student with name {name} & usn {usn} already exists in the database.")
            return

        # Create a new student object
        student = Student(
            usn=usn,
            name=name,
            course=course,
            year_join=year_join,
            attendance=0,
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


def print_all_student() -> None:
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


def get_all_student():
    students = session.query(Student).all()
    return students


def get_student(usn: int):
    try:
        student = session.query(Student).filter(Student.usn == usn).first()
        if not student:
            print(f"No student with usn {usn} found.")
            return
        return student
    except Exception as e:
        print(f"Error fetching details: {e}")


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


def return_tk_image(usn: int, scale: float = 0.3):
    try:
        student = session.query(Student).filter(Student.usn == usn).first()
        if student:
            student_image = Image.open(BytesIO(student.face_image))
            w, h = student_image.size
            size_scaled = (int(w * scale), int(h * scale))
            student_image = student_image.resize(size_scaled, Image.Resampling.LANCZOS)
            tk_image = ImageTk.PhotoImage(student_image)
            return tk_image
    except FileNotFoundError as e:
        print(f"Image file not found: {e}")
    except Exception as e:
        print(f"Error fetching image: {e}")


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


def update_credentials(usn: int, **kwargs):
    ''' Given the key value pairs of values, the credentials are updated '''
    student = session.query(Student).filter(Student.usn == usn).first()
    try:
        for attribute, new_val in kwargs.items():
            setattr(student, attribute, new_val)
            session.add(student)
            session.commit()
            print("Student credentials were updated successfully!")
    except Exception as e:
        print(f"Error in updating: {e}")


def delete_student(usn: int):
    student = session.query(Student).filter(Student.usn == usn).first()
    try:
        if student:
            session.delete(student)
            session.commit()
            print(f'Student {student.name} with {student.usn} deleted successfully.')
        else:
            print(f"No student with usn {usn} found in database.")
    except Exception as r:
        print(f"Error in deletion: {e}")


if __name__ == "__main__":
    # Testing
    try:
        create_student(400, "Samarth Sanjay Pyati", "B.Tech CSE", 2023, "F", "Male", "faces/400.jpg")
        create_student(87, "Atharv Bhujannavar", "B.Tech CSE", 2023, "I", "Male", "faces/087.jpeg")
        create_student(426, "Shashwath Jain H.P", "B.Tech CSE", 2023, "F", "Male", "faces/426.jpeg")
        create_student(418, "Sharan S Gowda", "B.Tech CSE", 2023, "I", "Male", "faces/418.jpeg")
        create_student(490, "Sushruth", "B.Tech CSE", 2023, "I", "Male", "faces/490.jpeg")
        create_student(540, "Vishnu Bhardhwaj", "B.Tech CSE", 2023, "I", "Male", "faces/540.jpeg")
        create_student(18, "Akhil Dayanand", "BBA Law", 2023, "A", "Male", "faces/018.jpeg")
        update_credentials(400, year_join=2023)
        delete_student(18)
        print_all_student()
    except Exception as e:
        print(f"Error occurred during testing: {e}")
