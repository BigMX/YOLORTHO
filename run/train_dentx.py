from ultralytics import YOLO

model = YOLO("./yolov8l-cus.yaml")  # build a new model from scratch
overrides = {"data": "./../data/dentx.yaml", 
            "model": "./yolov8l-cus.yaml",
            'task': 'custom',
             "imgsz": 1440, 
             "epochs": 150, 
             "workers": 1,
             "device": [6],
             "fliplr": 0.5,
             "flipud": 0.,
             "mosaic": 0.,
             "project": "dentx",
             "name": 'dentx',
             "save": True,
             'scale': 0.2,
             'shear': 0.,
             "batch": 2,
             }

cfg = './hyp.yaml'
model.train(cfg=cfg, **overrides
            )

# trainer = custom.CustomTrainer(overrides=overrides)
# trainer = segment.SegmentationTrainer(overrides=overrides)
# trainer.train()
# model.export(format="onnx")
# Use the model
# model.train(data="/home/pelle/yolov5/data/attachment.yaml",  device = [6,7], epochs=200)  # train the model