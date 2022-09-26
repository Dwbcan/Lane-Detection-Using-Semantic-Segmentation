import cv2
import numpy as np
import matplotlib.pyplot as plt

# Generate line coordinates based on line parameters
def generate_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(2.9/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

# Average the various detected lane lines into single line of best fit
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        
        # Fit a 1st degree polynomial to x and y points, returning vector of line parameters
        parameters = np.polyfit((x1, x2), (y1, y2), 1)

        slope = parameters[0]
        intercept = parameters[1]
        if slope > 0:
            right_fit.append((slope, intercept))
        else:
            left_fit.append((slope, intercept))
    right_fit_average = np.average(right_fit, axis=0)
    left_fit_average = np.average(left_fit, axis=0)
    right_line = generate_coordinates(image, right_fit_average)
    left_line = generate_coordinates(image, left_fit_average)
    return np.array([left_line, right_line])

# Perform Canny Edge Detection to detect edges in image
def canny_edge_detection(image):
    
    # Convert RGB image to grayscale 
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

# Display detected lane lines on image
def display_lane_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return line_image

# Isolate region of interest
def display_region_of_interest(image):
    height = image.shape[0]
    
    # Create triangle representing lane of interest
    polygons = np.array([[(250, height), (1100, height), (550, 250)]])
    
    # Turn image background black and highlight lane as white
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    
    # Compute bitwise_and to highlight only lane edges as white
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image



# The code below uses Lane Detection algorithm on JPEG image:

# Read in image of the road as a NumPy array of pixel intensities
image = cv2.imread('test_image.jpg')

lane_image = np.copy(image)
canny_image = canny_edge_detection(lane_image)
isolated_image = display_region_of_interest(canny_image)

# Perform Hough Transform to detect lines of best fit
lines = cv2.HoughLinesP(isolated_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

averaged_lines = average_slope_intercept(lane_image, lines)
line_image = display_lane_lines(lane_image, averaged_lines)

# Blend road image with image containing detected lane lines
final_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 0)

# Render final_image pixel array as an image that can be displayed
cv2.imshow('Output', final_image)

# Display image infinitely until a keyboard button is pressed
cv2.waitKey(0)



# # Uncomment the code below to use Lane Detection algorithm on a video:
# capture = cv2.VideoCapture('test2.mp4')
# while(capture.isOpened()):
#     _, frame = capture.read()
#     canny_image = canny_edge_detection(frame)
#     isolated_image = display_region_of_interest(canny_image)
#     lines = cv2.HoughLinesP(isolated_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
#     averaged_lines = average_slope_intercept(frame, lines)
#     line_image = display_lane_lines(frame, averaged_lines)
#     final_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
#     cv2.imshow('result', final_image)
#     if cv2.waitKey(1) == ord('q'):
#         break
# capture.release()
# cv2.destroyAllWindows()