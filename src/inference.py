from ultralytics import YOLO
import os

#data_folder = "/workspace/CAFSD/CAFSD/data.yaml"
model_pretrained = "/workspace/src/runs/detect/train/weights/best.pt"
pred_dir = '/workspace/src/predictions'
#save_results = "raid/amina_izbassar/"
# Load a model
model = YOLO(model_pretrained)  # build a new model from scratch

# Use the model
#results = model.train(data=data_folder, imgsz=640, epochs=50, augment=True, device=[0]) #, hsv_h=0.5, hsv_s=0.5, translate=1, flipud=1, fliplr=1, mixup=0.5, degrees=0.5)  # train the model

# results = model.val()  # evaluate model performance on the validation set

#imgs = os.listdir('five/check/images')
#for img in imgs:
img_path = '/workspace/CAFSD/CAFSD/CAFSD/test/images/10010_jpg.rf.40292fc26b1458a62aa071e9b8b35cc7.jpg'
results = model.predict(img_path, conf=0.3)[0]
#import ipdb
#ipdb.set_trace()
for i, result in enumerate(results):
    if len(result.boxes.cls) != 0:
        description_file = os.path.join(pred_dir, os.path.splitext(os.path.basename(img_path))[0]+'.txt')
        print(description_file)
        with open(description_file, 'a+') as f:
            f.write(str(result.boxes.cls.item()))
            for x in result.boxes.xywh[0]:
                f.write(' ')
                f.write(str(x.item()))
            f.write('\n')
#results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
#results = model.export(format="onnx")  # export the model to ONNX format
#validation_results = model.val()
