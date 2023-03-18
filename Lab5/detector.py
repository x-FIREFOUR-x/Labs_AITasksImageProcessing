from imageai.Detection import ObjectDetection


model_path = "models/yolo-tiny.h5"
input_path = "data/input/test_im.jpg"
output_path = "data/output/test_im.jpg"

detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()

detector.setModelPath(model_path)
detector.loadModel()
detection = detector.detectObjectsFromImage(
    input_image=input_path, output_image_path=output_path)

for eachItem in detection:
    print(f"{eachItem['name']}:{eachItem['percentage_probability']}")
