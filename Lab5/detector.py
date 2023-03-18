from imageai.Detection import ObjectDetection

model_path = "models/yolo-tiny.h5"
detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()

detector.setModelPath(model_path)
detector.loadModel()

def detect_objects(input_path, output_path, print_object_flg=False):
    detection = detector.detectObjectsFromImage(
        input_image=input_path, output_image_path=output_path)

    if print_object_flg:
        for eachItem in detection:
            print(f"{eachItem['name']}:{eachItem['percentage_probability']}")
