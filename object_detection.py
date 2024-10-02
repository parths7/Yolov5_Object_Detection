import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import cv2
import torch


class ObjectDetection():

  def __init__(self, capture = 0):
    self.capture = capture
    self.model = self.load_model()
    self.classes = self.model.names


  def load_model(self):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

  def predict(self, img):
    results = self.model(img)
    return results

  def plot_boxes(self, results, img):
    for box in results.xyxy[0]:
      x1, y1, x2, y2, conf, cls = box
      x1, y1, x2, y2, conf, cls = int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)

      current_class = self.classes[cls]

      if conf > 0.3:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{current_class} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0,255), 2)

    return img

  def __call__(self):
    cap = cv2.VideoCapture(self.capture)
    assert cap.isOpened()

    while True:

      ret, frame = cap.read()
      assert ret

      results = self.predict(frame)
      frame = self.plot_boxes(results, frame)

      cv2.imshow('Image', frame)
      if cv2.waitKey(1) == ord("q"):
        break

    cap.relase()
    cv2.destroyAllWindows()
