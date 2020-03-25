# import the necessary packages
from imutils.video import FPS
import numpy as np
import imutils
import cv2

net = cv2.dnn.readNetFromCaffe("ali.prototxt.txt", "res10_ali_ssd_140kiter.caffemodel")

# initialize the video file 
# if you want to use webcam then use the following commands
# vid_stream = VideoStream(src=0).start()
# time.sleep(2.0)
print("The video stream is being started ...")
vid_stream = cv2.VideoCapture("RDJ1.mp4")
fps = FPS().start()

# loop over the frames from the video stream
while True:
    # acquire the frames from the video file
	(acquired, frm) = vid_stream.read()
	# if the frame was not acquired, then video is ended
	if not acquired:
		break
	# acquire the frame from the video file and resize it
	# to width of 400 pixels
	frm = imutils.resize(frm, width=400)
 
	# grab the frame dimensions and convert it to a bounding box
	(ht, wt) = frm.shape[:2]
	bbox = cv2.dnn.blobFromImage(cv2.resize(frm, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
 
	# pass the bounding box through the network and obtain the objects and
	# predictions
	net.setInput(bbox)
	objs = net.forward()

	# loop over the objects/faces
	for i in range(0, objs.shape[2]):
		# extract the probabilities associated with the
		# prediction
		prob = objs[0, 0, i, 2]

		# Use the threshold to detect the faces/objs
	     # greater than the minimum probability
		if prob < 0.4:
			continue

		# compute the (x, y)-coordinates of the bounding box for thee
		# object
		rect = objs[0, 0, i, 3:7] * np.array([wt, ht, wt, ht])
		(Xs, Ys, Xe, Ye) = rect.astype("int")
 
		# draw the bounding box of the face along with the associated
		# probability
		text = "{:.2f}%".format(prob * 100)
		G = Ys - 10 if Ys - 10 > 10 else Ys + 10
		cv2.rectangle(frm, (Xs, Ys), (Xe, Ye),
			(0, 0, 255), 2)
		cv2.putText(frm, text, (Xs, G),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# show the output frame
	cv2.imshow("Frame", frm)
	key = cv2.waitKey(1) & 0xFF

# do a bit of cleanup
vid_stream.release()
cv2.destroyAllWindows()