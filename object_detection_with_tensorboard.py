import os
import cv2
import time
import numpy as np
import tensorflow as tf

from utils import FPS, WebcamVideoStream
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from PIL import Image
import requests
from io import BytesIO

CWD_PATH = os.getcwd()
TF_MODELS_PATH = os.path.join(CWD_PATH, '..', 'TensorFlow Object Detection Models', 'trained_models')
# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
"""Models available: 
    ssd_mobilenet_v1_coco_11_06_2017 -> Fast - 21 mAP
    ssd_inception_v2_coco_11_06_2017 -> Fast - 24 mAP
    rfcn_resnet101_coco_11_06_2017 -> Medium - 30 mAP
    faster_rcnn_resnet101_coco_11_06_2017 -> Medium - 32 mAP
    faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017 -> Slow - 37 mAP
"""
PATH_TO_CKPT = os.path.join(TF_MODELS_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def model_load_into_memory():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    
    # Draw just class 1 detections (Person)
    classes_np = np.squeeze(classes).astype(np.int32)
    print(classes_np)
    scores_np = np.squeeze(scores)
    print("Scores 1\n")
    print(scores_np)
    total_people = 0 # Total observations per frame
    for i in range(classes_np.size):
        if classes_np[i]==1 and scores_np[i]>=0.5:
            total_people += 1
        elif classes_np[i] != 1:
            scores_np[i] = 0.02
    print("######################### " + str(total_people) + " ########################")
    print("Scores 2\n")
    print(scores_np)

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        scores_np,
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    return image_np

def main():
    fps = FPS().start()
    
    detection_graph = model_load_into_memory()
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            while True:  # fps._numFrames < 120
                #frame = video_capture.read()
                #response = requests.get('http://14.141.75.148/cgi-bin/camera?resolution=640&amp;amp;quality=1&amp;amp;Language=0&amp;amp;1501245077') # Ídolo hindú
                #response = requests.get('http://203.138.220.33/mjpg/video.mjpg') # Calles de Osaka
                response = requests.get('http://31.168.54.91/cgi-bin/camera?resolution=640&amp;quality=1&amp;Language=0&amp;COUNTER') # Tienda ropa en Tel Aviv
                        
                frame = np.array(Image.open(BytesIO(response.content)))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imshow('Entrada', frame)
                
                t = time.time()
            
                output = detect_objects(frame, sess, detection_graph)
        
                cv2.imshow('Salida', output)

                fps.update()
        
                print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))
        
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
            fps.stop()
            print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
            print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
            
            with tf.name_scope('Wx_plus_b'):
                aaa = tf.constant([5, 6])
                tf.summary.scalar('aaa', aaa)
            with tf.name_scope('Imagen'):
                tf.summary.image('Salida', output)
            merged = tf.summary.merge_all()
            file_writer = tf.summary.FileWriter('TensorBoard logs', sess.graph)
            file_writer.add_graph(sess.graph)
        
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    