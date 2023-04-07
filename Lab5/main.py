import cv2
import imutils

from detector import detect_objects


def detect_in_image(input_path, output_path, scale = 1):
    detect_objects(input_path, output_path)

    img = cv2.imread(output_path)
    h, w = img.shape[0:2]
    img = imutils.resize(img, height=h * scale)

    cv2.imshow("Image", img)
    cv2.waitKey(0)


def detect_in_video(input_path):
    video_capture = cv2.VideoCapture(input_path)
    input_buf = "data/buffer/buf.jpg"
    output_buf = "data/buffer/buf.jpg"

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        cv2.imwrite(input_buf, frame)
        if ret:
            detect_objects(input_buf, output_buf)
            frame = cv2.imread(output_buf)
            cv2.imshow("video", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    #detect_in_image("data/input/test_im.jpg", "data/output/test_im.jpg")
    detect_in_image("data/input/fruit2.jpeg", "data/output/fruit2.jpg", 2)
    detect_in_video("data/input/video.mp4")
