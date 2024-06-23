from ultralytics import YOLO
import os

data_folder = "/workspace/sata/CAFSD/CAFSD/data.yaml"
model_pretrained = "/workspace/yolov8n.pt"
#save_results = "raid/amina_izbassar/"
# Load a model
model = YOLO(model_pretrained)  # build a new model from scratch

# Use the model
results = model.train(data=data_folder, imgsz=640, epochs=50, augment=True, device=[0]) #, hsv_h=0.5, hsv_s=0.5, translate=1, flipud=1, fliplr=1, mixup=0.5, degrees=0.5)  # train the model

# results = model.val()  # evaluate model performance on the validation set

#imgs = os.listdir('five/check/images')
#for img in imgs:
#    model.predict(os.path.join('five/check/images', img), save=True, imgsz=320, conf=0.3)
#results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
results = model.export(format="onnx")  # export the model to ONNX format
#validation_results = model.val()
