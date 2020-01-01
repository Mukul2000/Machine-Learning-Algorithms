import cv2
import time
import numpy as np
from keras.preprocessing import image
import tensorflow as tf

model = tf.keras.models.load_model('HandWrittenDigits_Conv.h5')


cv2.namedWindow("Digit Detection")
vc = cv2.VideoCapture(0)
vc.set(cv2.CAP_PROP_FRAME_WIDTH, 28)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 28)

if(vc.isOpened()):
	rval, frame = vc.read()
else:
	rval = False

while rval:
	cv2.imshow("Digit Detection", frame)
	rval,frame = vc.read()
	#cv2.resize(frame,(28,28))
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	img = image.img_to_array(frame)
	img_tensor = np.expand_dims(img,axis=0)
	img_tensor /= 255
	img_tensor = tf.image.resize(img_tensor, (28,28))
	pred = model.predict(img_tensor,steps = 1)
	#x = np.where(pred >= 0.5)
	#print(pred[x])
	print(pred)

	time.sleep(0.100)

	key = cv2.waitKey(20)
	if(key == 27):
		break

cv2.destroyWindow("Digit Detection")
vc.release()