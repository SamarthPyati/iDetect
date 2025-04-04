from faceRecUtils import FaceRecognition
from time import perf_counter

if __name__ == "__main__":
    tic = perf_counter()
    fr = FaceRecognition()
    fr.run_recognition()
    toc = perf_counter()

    elapsed = toc - tic
    print(f"Total Execution Time: {elapsed:.2f} second(s)")
