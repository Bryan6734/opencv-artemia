
"""
artemia-opencv
By Bryan Sukidi
"""

import cv2
import matplotlib.pyplot as plt

# Below, I initialized some of the boilerplate needed to open the video and display windows.
window_name = "Brine Shrimp Camera"
cv2.namedWindow(window_name)

video_path = "purple-red/purple-red-5.mov"
video = cv2.VideoCapture(video_path)

# I keep track of 3 different lists of data which are updated in the main loop.
# Using these lists, I created a live plot of the data using matplotlib. 
left_shrimp_record = []
right_shrimp_record = []
total_shrimp = []

fig, ax = plt.subplots()

ax.set_xlim(0, 500)
ax.set_ylim(0, 40)

left_shrimp_line, = ax.plot([], [], label='Left Shrimp')
right_shrimp_line, = ax.plot([], [], label='Right Shrimp')
total_shrimp_line, = ax.plot([], [], label='Max Shrimp')


ax.legend()

plt.title("Brine Shrimp Distribution")
plt.xlabel("Frames Processed")
plt.ylabel("Number of Shrimp")
plt.show(block=False)

"""
The MOG2 algorithm is a process that essentially subtracts the background from the foreground.
It essentially estimates the intensity distribution of the background and then uses that to
determine the foreground. When you do this a number of times, you can get a pretty good idea of 
what the foreground is based off what intensities are changing a LOT vs a LITTLE from frame to frame.
"""
backgroundSubtractor = cv2.createBackgroundSubtractorMOG2()

while True:

    ret, frame = video.read()
    window_width = cv2.getWindowImageRect(window_name)[2]

    foreground = backgroundSubtractor.apply(frame)

    """
    The GaussianBlur function takes in a kernel size and uses that to blur the image.
    Blurring helps us remove noise from the image and make it easier to find contours.
    The (23, 23) in this case is the value I got after testing a bunch of different values,
    and it works by taking the average of the 23x23 pixel square around each pixel and calculating
    pixel intensity based off that average (thereby acheiving a blurring effect).
    """
    foreground = cv2.GaussianBlur(foreground, (23, 23), 0)

    # After manually testing values from 0-255, I found that 50 was a good threshold value.
    # Because we've blurred the image, now each pixel is a different value (intensity) between 0-255
    # (0 being black and 255 being white). The threshold function essentially converts the image to
    # black and white based off a threshold value. In ours, IF the intensity is greater than 50, we'll change it to 255,
    # else if it's less than 50, we'll change it to 0. This sharpens the image a bit.
    _, thresh = cv2.threshold(foreground, 50, 255, cv2.THRESH_BINARY)

    # The findContours function finds the contours in the image. Contours are essentially the outlines of objects
    # and it's a function that comes with OpenCV. The RETR_EXTERNAL parameter tells the function to only find the
    # outermost contours (which is what we want). The CHAIN_APPROX_SIMPLE parameter tells the function to only
    # return the endpoints of the contours (which is also what we want).
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Here, I made two simple variables for counting the shrimp on the left and right side.
    left_shrimp = 0
    right_shrimp = 0

    for contour in contours:

        # To get rid of even more noise, I calculated the area of each "blob" in our image in pixels. If the area
        # is less than 100 pixels, we'll ignore it since it's probably just noise (and almost all of the shrimp are bigger than this).
        # If it's greater than 100 pixels, we'll draw a circle around it and add it to the left or right shrimp count.
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

    # The code below is just for updating the lat matplotlib plot. 
    # I'm keeping track of the number of shrimp on the left, right, and the total.
    left_shrimp_record.append(left_shrimp)
    right_shrimp_record.append(right_shrimp)
    total_shrimp.append(left_shrimp + right_shrimp) 

    left_shrimp_line.set_data(range(len(left_shrimp_record)), left_shrimp_record)
    right_shrimp_line.set_data(range(len(right_shrimp_record)), right_shrimp_record)
    total_shrimp_line.set_data(range(len(total_shrimp)), total_shrimp)
    fig.canvas.draw()

    cv2.putText(frame, f"Left Shrimp: {left_shrimp}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Right Shrimp: {right_shrimp}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Here, I displayed each mask and the original frame. You can see the different steps we went through.
    # The foreground mask applies the background subtractor (MOG2) and the Gaussian blur.
    # The threshold mask takes the blurred image and filters it based off the "bright" pixels.
    # The original frame is just the original frame from the video. We use the threshold mask to calculate
    # the contours, but we draw everything on the Brine Shrimp camera. 
    cv2.imshow("Foreground Mask", foreground)
    cv2.imshow("Thresholded Mask", thresh)
    cv2.imshow("Brine Shrimp Camera", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break


video.release()