import cv2
import numpy

def found_faces(path_image, scaling_factor):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    frame = cv2.imread(path_image)
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

    print(f"Found {len(face_rects)} faces")
    for (x, y, w, h) in face_rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)

    cv2.imshow("Image", frame)
    cv2.waitKey(0)


def found_smile_eye_faces(path_image, scaling_factor):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    frame = cv2.imread(path_image)
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
    print(f"Found {len(face_rects)} faces")

    for (x, y, w, h) in face_rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)

        roi_gray = frame_gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        smile = smile_cascade.detectMultiScale(roi_gray)
        eye = eye_cascade.detectMultiScale(roi_gray)

        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)
        for (ex, ey, ew, eh) in eye:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)

    cv2.imshow("Image", frame)
    cv2.waitKey(0)

def found_faces_video(path_video):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cv2.startWindowThread()
    video = cv2.VideoCapture(path_video)
    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0XFF==ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def found_people(path_video):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cv2.startWindowThread()
    video = cv2.VideoCapture(path_video)
    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)

        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))
        boxes = numpy.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        for (xa, ya, xb, yb) in boxes:
            cv2.rectangle(frame, (xa, ya), (xb, yb), (0, 255, 255), 1)

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    found_smile_eye_faces("media\\1.jfif", 3)
    found_smile_eye_faces("media\\2.jfif", 4)

    found_faces_video("media\\video.mp4")

    found_people("media\\video2.mp4")