# import cv2
# import numpy as np
# # import the video 

# # video_path = "IMG_8717.mov"
# video_path = "output_video.mp4"
# video = cv2.VideoCapture(video_path)




# # backgroundSubtractor = cv2.createBackgroundSubtractorKNN()
# backgroundSubtractor = cv2.createBackgroundSubtractorMOG2()

# # kalman = cv2.KalmanFilter(4, 2)
# # kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], np.float32)
# # kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
# # kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
# # kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.1
# # kalman.errorCovPost = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)


# """
# MOG2 -> models each pixel by a mixture of gaussian distributions
# -- when a new frame is processed, mog2 compares the pixel instensity with the gaussian distribution
# -- gaussian distribution: a bell curve that represents the probability of a pixel intensity (0-255) using the mean and the standard deviation
# -- if the pixel intensity is far from the gaussian distribution, it is considered foreground
# -- if the pixel intensity is close to the gaussian distribution, it is considered background

# KNN -> models each pixel by selecting K nearest neighbors
# -- compares pixel intensity with neighbors and decides if it is foreground or background
# """

# while True:
#     ret, frame = video.read()
#     # ret is a boolean (is frame available or no?)
#     # frame is a numpy pixel array. first 2 dimensions are height and width,
#     # and the last dimension is the color channel (BGR)



#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     mask = backgroundSubtractor.apply(gray)

#     # add a blur to the mask
#     mask = cv2.GaussianBlur(mask, (5, 5), 0)


#     # erode the mask
#     # erode(inputimg, kernel, iterations)
#     # basically allows us to remove small noise pixels
#     mask = cv2.erode(mask, None, iterations=3)

#     mask = cv2.dilate(mask, None, iterations=5)

#     # dilate the mask
#     # dilate(inputimg, kernel, iterations)
#     # basically allows us to fill in holes

#     # threshold the mask
#     # threshold(inputimg, thresholdvalue, maxvalue, thresholdtype)
#     # for the last param, there are different types of thresholding (binary, inversion, 0 for all under threshold, etc)
#     _, mask = cv2.threshold(mask, 140, 255, cv2.THRESH_BINARY)

#     # find contours
#     # a contour is a curve joining all continuous points along the boundary of an object (almost like perimeter)
#     # findContours(inputimg, retrievalmode, approximationmethod)
#     # retrieval methods:
#     # --- RETR_EXTERNAL: retrieves only the extreme outer contours (no nesting or child contours)
#     # --- RETR_LIST: retrieves all contours
#     # --- RETR_TREE: retrieves all contours and creates a full family hierarchy list

#     # approximation methods:
#     # --- CHAIN_APPROX_NONE: stores all contour points
#     # --- CHAIN_APPROX_SIMPLE: compresses horizontal, vertical, and diagonal segments and leaves only their end points

#     contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # for contour in contours[0]:
#     #     # drawContours(inputimg, contours, contourindex, color, thickness)
#     #     cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
#     #     # -1 means draw all contours

#     # # draw all contours to frame

#     # draw lines between all points in the contour
#     # for contour in contours[0]:
#     #     for i in range(len(contour) - 1):
#     #         cv2.line(frame, tuple(contour[i][0]), tuple(contour[i + 1][0]), (0, 255, 0), 2)

#     # draw bounding rectangles around contours

#     window_width = cv2.getWindowImageRect(window_name)[2]
#     left_shrimp = 0
#     right_shrimp = 0

#     predicted_position = None

#     # loop over the contours
#     for contour in contours[0]:

#         # # compute the centroid of the contour
#         M = cv2.moments(contour)
#         cx = int(M['m10'] / M['m00'])
#         cy = int(M['m01'] / M['m00'])

#         # # predict the position of the object using the Kalman filter
#         # if predicted_position is None:
#         #     kalman.statePost[0] = cx
#         #     kalman.statePost[2] = cy
#         #     predicted_position = (cx, cy)
#         # else:
#         #     kalman.predict()
#         #     predicted_position = (int(kalman.statePost[0]), int(kalman.statePost[2]))

#         # # update the position of the object using the Kalman filter
#         # z = np.array([[cx], [cy]], dtype=np.float32)
#         # kalman.correct(z)
#         # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)


#         if cx > window_width / 2:
#             right_shrimp += 1
#         else:
#             left_shrimp += 1

#         cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

#     # for contour in contours[0]:
#         # # boundingRect(contour) returns (x, y, w, h)
#         # x, y, w, h = cv2.boundingRect(contour)
#         # if x < window_width / 2:
#         #     left_shrimp += 1
#         # else:
#         #     right_shrimp += 1

#         # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

#     cv2.putText(frame, f"Left Shrimp: {left_shrimp}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv2.putText(frame, f"Right Shrimp: {right_shrimp}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


#     # shows frame
#     cv2.imshow("Brine Shrimp Camera", frame)
#     cv2.imshow("Mask", mask)
#     # wait for key press every 1 millisecond
#     key = cv2.waitKey(1)

#     # if q is pressed, exit
#     if key == ord('q'):
#         break


# video.release()
# video.destroyAllWindows()



import cv2
import matplotlib.pyplot as plt

window_name = "Brine Shrimp Camera"
cv2.namedWindow(window_name)


# create a background subtractor
backSub = cv2.createBackgroundSubtractorMOG2()

# create a kernel for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# open the video stream
video = cv2.VideoCapture("purple-red/purple-red-5.mov")

left_shrimp_record = []
right_shrimp_record = []
total_shrimp = []

# create a figure and axis object
fig, ax = plt.subplots()

# set the x and y limits of the plot
ax.set_xlim(0, 500)
ax.set_ylim(0, 40)

# create line objects to plot the data
line1, = ax.plot([], [], label='Left Shrimp')
line2, = ax.plot([], [], label='Right Shrimp')
line3, = ax.plot([], [], label='Max Shrimp')



# add a legend to the plot
ax.legend()
# show the plot

plt.xlabel("Frames Processed")
plt.ylabel("Number of Shrimp")
plt.show(block=False)


while True:
    # read a frame from the video stream
    ret, frame = video.read()

    window_width = cv2.getWindowImageRect(window_name)[2]

    # apply background subtraction to obtain the foreground mask
    fgMask = backSub.apply(frame)
    fgMask = cv2.GaussianBlur(fgMask, (23, 23), 0)

    # apply adaptive thresholding to binarize the image
    # thresh = cv2.adaptiveThreshold(fgMask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _, thresh = cv2.threshold(fgMask, 50, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    left_shrimp = 0
    right_shrimp = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:

            (x, y), radius = cv2.minEnclosingCircle(contour)

            if x > window_width / 2:
                right_shrimp += 1
            else:
                left_shrimp += 1
            center = (int(x), int(y))
            cv2.circle(frame, center, 4, (0, 255, 0, 10), -1)
            cv2.putText(frame, str(area), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 0, bottomLeftOrigin=False)

    left_shrimp_record.append(left_shrimp)
    right_shrimp_record.append(right_shrimp)
    total_shrimp.append(left_shrimp + right_shrimp) 

    line1.set_data(range(len(left_shrimp_record)), left_shrimp_record)
    line2.set_data(range(len(right_shrimp_record)), right_shrimp_record)
    line3.set_data(range(len(total_shrimp)), total_shrimp)
    fig.canvas.draw()

    # pause for a short time
    plt.pause(0.01)

    # count the number of brine shrimp on the left vs. the right of the video


    cv2.putText(frame, f"Left Shrimp: {left_shrimp}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Right Shrimp: {right_shrimp}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # show the output
    # cv2.imshow("Foreground Mask", fgMask)
    # cv2.imshow("Thresholded Image", thresh)
    cv2.imshow("Brine Shrimp Camera", frame)
    # cv2.imshow("Opening", opening)

    # wait for key press every 1 millisecond
    key = cv2.waitKey(1)

    # if q is pressed, exit
    if key == ord('q'):
        break









video.release()
cv2.destroyAllWindows()