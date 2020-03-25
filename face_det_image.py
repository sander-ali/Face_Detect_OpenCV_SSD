
# import the necessary packages
import numpy as np
import cv2

net = cv2.dnn.readNetFromCaffe("ali.prototxt.txt", "res10_ali_ssd_140kiter.caffemodel")

# load the input image and construct an input rectangle for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
img = cv2.imread("Sander1.jpg")
(ht, wt) = img.shape[:2]
bbox = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))

# pass the rectangle through the network and obtain the detections and
# predictions
print("Detected Objects are being computed, wait ....")
net.setInput(bbox)
objs = net.forward()

# loop over the objs
for i in range(0, objs.shape[2]):
	# extract the probability associated with the
	# prediction
	prob = objs[0, 0, i, 2]

	# Use the threshold to detect the faces/objs
	# greater than the minimum probability
	if prob > 0.5:
		# compute the (x, y)-coordinates of the bounding box for the
		# object
		rect = objs[0, 0, i, 3:7] * np.array([wt, ht, wt, ht])
		(Xs, Ys, Xe, Ye) = rect.astype("int")
 
		# draw the bounding box of the face along with the associated
		# probability
		text = "{:.2f}%".format(prob * 100)
		G = Ys - 10 if Ys - 10 > 10 else Ys + 10
		cv2.rectangle(img, (Xs, Ys), (Xe, Ye),
			(0, 0, 255), 2)
		cv2.putText(img, text, (Xs, G),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show the output image
cv2.imshow("Output", img)
cv2.waitKey(0)
