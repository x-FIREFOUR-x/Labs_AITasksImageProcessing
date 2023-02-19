import cv2
import imutils
import numpy as np

if __name__ == '__main__':
    #2
    img1 = cv2.imread("image.jpg")
    cv2.imshow("Image", img1)
    cv2.waitKey()

    img2 = cv2.imread("image.jpg", 0)
    cv2.imshow("Image", img2)
    cv2.waitKey()

    cv2.imwrite("image_1.jpg", img2)
   
    #3
    img3 = cv2.imread("image.jpg")
    (blue, green, red) = img3[100, 50]
    print(f"B: {blue}, R: {red}, G: {green}")

    #4
    img = cv2.imread("image.jpg")
    crop = img[240:500, 240:500]
    cv2.imshow("Cropped Image", crop)
    cv2.waitKey()
    
    #5
    img = cv2.imread("image.jpg")
    resizes = cv2.resize(img, (800, 200))
    cv2.imshow("Resized Image", resizes)
    cv2.waitKey()
    
    #6
    img = cv2.imread("image.jpg")
    h, w = img.shape[0:2]
    h_new = 400
    w_new = int(h_new * w / h)
    resized = cv2.resize(img, (w_new, h_new))
    cv2.imshow("Resized Image", resized)
    cv2.waitKey()

    img = cv2.imread("image.jpg")
    resized = imutils.resize(img, height=400)
    cv2.imshow("Resized Image", resized)
    cv2.waitKey()
    

    #7
    img = cv2.imread("image.jpg")
    h, w = img.shape[0:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -45, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    cv2.imshow("Rotated Image", rotated)
    cv2.waitKey()

    img = cv2.imread("image.jpg")
    rotated = imutils.rotate(img, -45)
    cv2.imshow("Rotated Image", rotated)
    cv2.waitKey()
    

    #8
    img = cv2.imread("image.jpg")
    resized = imutils.resize(img, height=400)
    blurred = cv2.GaussianBlur(resized, (11, 11), 0)
    summing = np.hstack((resized, blurred))
    cv2.imshow("Blurred Image", summing)
    cv2.waitKey()
    

    #9
    img = cv2.imread("image.jpg")
    cv2.rectangle(img, (0, 0), (150, 150), (255, 0, 0), 2)
    cv2.imshow("Rectangle", img)
    cv2.waitKey()

    img = cv2.imread("image.jpg")
    cv2.line(img, (0, 0), (150, 150), (255, 0, 0), 2)
    cv2.imshow("Line", img)
    cv2.waitKey()

    img = cv2.imread("image.jpg")
    cv2.circle(img, (150, 150), 50, (255, 0, 0), 2)
    cv2.imshow("Circle", img)
    cv2.waitKey()

    img = cv2.imread("image.jpg")
    points = np.array([[0, 0], [150, 120], [120, 150], [0, 0]])
    cv2.polylines(img, np.int32([points]), 1, (255, 0, 0))
    cv2.imshow("PolyLines", img)
    cv2.waitKey()

    #10
    img = cv2.imread("image.jpg")
    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    cv2.putText(img, "OpenCV", (100, 300), font, 4, (120, 255, 120), 4, cv2.LINE_4)
    points = np.array([[0, 0], [150, 120], [120, 150], [0, 0]])
    cv2.imshow("Text", img)
    cv2.waitKey()

