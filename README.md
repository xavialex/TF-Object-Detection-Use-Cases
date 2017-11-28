# Object Detection Use Cases

Deep Learning (DL) has achieved huge success in the last years in the field of Computer Vision. Researchers around the world work in the building and training of new state-of-the-art architectures and models. This requires lots of knowledge, skill, data an computing power, who may be hard to get for the vast majority of DL practitioners. That's the main reason because Transfer Learning becomes a very good approach in order to get introduced in this field, which consists of grabbing some architecture/models weights that have demonstrated their value already in well known image datasets and build new applications starting from there.
In this repository appears some apps with different use cases that take use of TensorFlow's Object Detection Models Zoo that may be of interest for those of who require strong Object Detection capabilities.

## Dependencies

It's mandatory to download a [Tensorflow detection model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and locate it like this:
```
├───TensorFlow Object Detection Models
│   └───trained_models
│       ├───faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017
│       │
│       ├───faster_rcnn_resnet101_coco_11_06_2017
│       │
│       ├───rfcn_resnet101_coco_11_06_2017
│       │
│       ├───ssd_inception_v2_coco_11_06_2017
│       │
│       ├───ssd_mobilenet_v1_coco_11_06_2017
│       │
│       └───...
│
├───TF-Object-Detection-Use-Cases
│   │   object_detection_from_url.py
│   │   object_detection_from_video.py
│   │   object_detection_from_webcam.py
│   │   object_detection_with_tensorboard.py
│   │   people_detection_from_webcam.py
│   │   utils.py
│   │   ...
│   │
│   └───object_detection
```

Depending on the specs of your environment, you may choose one faster or other wiht a better mean Average Precision (mAP).

Also, depending on the example you want to run you'll need to download certain libraries or have access to hardware (webcam).

## Use

Launch any of the *.py* examples from your favourite IDE or from the command prompt. Press the *'q'* button in the resulting floating window to close the program.

Depending on the specs of your environment, you may choose one faster or other wiht a better mean Average Precision (mAP).

Also, depending on the example you want to run you'll need to download certain libraries or have access to hardware (webcam).
