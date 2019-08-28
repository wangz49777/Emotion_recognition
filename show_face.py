import tensorflow as tf
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
emo_labels = ['angry', 'disgust:', 'fear', 'happy', 'sad', 'surprise', 'neutral']
emo_new = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown']
num_class = len(emo_labels)

testmodel_path = './testmodel/'
default_size = 48
img_size = 48
channel = 1
ckpt_name = 'emotion_cnn.ckpt'
ckpt_path = os.path.join(testmodel_path, ckpt_name)
cascade_path = os.path.join(testmodel_path, 'haarcascade_frontalface_alt.xml')

gpu_options = tf.GPUOptions(allow_growth = True)
config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True)
sess = tf.Session(config = config)
saver = tf.train.import_meta_graph(ckpt_path + '.meta')
saver.restore(sess, ckpt_path)
graph = tf.get_default_graph()

x_input = graph.get_tensor_by_name('input/x_input:0')
dropout = graph.get_tensor_by_name('hyperparameters/dropout:0')
softmax = graph.get_tensor_by_name('evaluate/softmax:0')
color = (0, 0, 255)
cap = cv2.VideoCapture(0)

while True:
    b, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cascade_path)
    faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.1, minNeighbors = 1, minSize = (120, 120))
    if len(faceRects) > 0:
        for face in faceRects:
            x, y, w, h = face
            images=[]
            # rs_sum=np.array([0.0]*num_class)
            image = frame_gray[y: y + h, x: x + w ]
            image=cv2.resize(image,(img_size,img_size))
            image=image*(1./255)
            images.append(image)
            images.append(cv2.flip(image,1))
            images.append(cv2.resize(image[2:45,:],(img_size,img_size)))
            images = np.reshape(images,[3,48,48,1])
            
            softmax_ = sess.run(softmax, {x_input: images, dropout: 1.0})
            softsum = np.sum(softmax_, axis= 0)
#             class_ = np.argmax(softsum, axis= 1)
            class_ = np.argmax(softsum)
            print(class_)
            emo = emo_new[class_]
#             print ('Emotion : ',emo)
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,'%s' % emo,(x + 30, y + 30), font, 1, (255,0,255),4)
        cv2.imshow("me", frame)
    if cv2.waitKey(100) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()