import cv2
import torch
import colorsys
import numpy as np
import time

class YOLOV5(object):

    def __init__(self, name: str = None):
        if name:
            self.name_model = name
        else:
            self.name_model = 'yolov5s'
        self.model = torch.hub.load('ultralytics/yolov5', self.name_model, pretrained=True)
        self.names = dict(zip(self.model.names.values(), self.model.names.keys()))
        hsv_tuples = [(x / len(self.model.names.values()), 1., 1.)
                      for x in range(len(self.model.names.values()))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))


    def _detectionImage(self, image: np.array):

        return self.model(image).pandas().xyxy[0]

    def getDetectionImage(self, image: np.array):

        result = self._detectionImage(image)

        thickness = (image.shape[0] + image.shape[1]) // 600

        classes = result.name.unique()

        for cls in classes:
            box = result.loc[result.name==cls].values.tolist()[0][:4]
            score = result.loc[result.name==cls].values.tolist()[0][4]

            label = '{} {:.2f}'.format(cls, score)
            scores = '{:.2f}'.format(score)

            left, bottom,  right, top = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))

            cv2.rectangle(image, (left, top), (right, bottom), self.colors[self.names[cls]], thickness)

            #text size
            (test_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                                  thickness / 1, 1)

            #text rectangle
            cv2.rectangle(image, (left, top), (left + test_width, top - text_height - baseline), self.colors[self.names[cls]],
                          thickness=cv2.FILLED)

            cv2.putText(image, label, (left, top - 2), cv2.FONT_HERSHEY_SIMPLEX, thickness / 1, (0, 0, 0),
                        1)

        return image

if __name__ == '__main__':

    yolo = YOLOV5()

    start_time = time.time()
    display_time = 2
    fps = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("We cannot open webcam")

    while True:
        ret, frame = cap.read()
        r_image = yolo.getDetectionImage(frame)

        # show us frame with detection
        cv2.imshow("Web cam input", r_image)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

        fps += 1
        TIME = time.time() - start_time
        if TIME > display_time:
            print("FPS:", fps / TIME)
            fps = 0
            start_time = time.time()

    cap.release()
    cv2.destroyAllWindows()
