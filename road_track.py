import cv2
import numpy as np
import matplotlib.pyplot as plt


image_path = r'C:\Users\Kazi\Desktop\exclude\image_process\test_image.jpg'
video_path = r"C:\Users\Kazi\Desktop\exclude\image_process\test2.mp4"

def average_slope_intercept(image,lines):
	left_fit= []
	right_fit =[]
	for line in lines:
		x1,y1,x2,y2 = line.reshape(-1)
		parameters = np.polyfit((x1,x2),(y1,y2),1)
		slope = parameters[0]
		intercept = parameters[1]
		if slope > 0 :
			right_fit.append((slope,intercept))
		else:
			left_fit.append((slope,intercept))
	left_fit_average = np.average(left_fit,axis=0)
	right_fit_average = np.average(right_fit,axis=0)
	left_line = make_coordinates(image,left_fit_average)
	right_line = make_coordinates(image,right_fit_average)
	return np.array([left_line,right_line])
    

def make_coordinates(image,line_parameters):
	slope = line_parameters[0]
	intercept=line_parameters[1]
	y1 = image.shape[0]
	print(line_parameters)
	y2 = int(y1*(3/5))
	x1 = int((y1-intercept)/slope)
	x2 = int((y2-intercept)/slope)
	return np.array([x1,y1,x2,y2])

    	
def canny(image):
	gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5),0)
	canny = cv2.Canny(blur,50,150)
	return canny


def region_of_interest(image):
	height = image.shape[0]
	mask = np.zeros_like(image)
	triangle =np.array([
		[(200,height),(1100,height),(550,250)]
	])
	cv2.fillPoly(mask,triangle,255)
	masked_image = cv2.bitwise_and(image,mask)
	return masked_image

def display_line(image,lines):
	line_image = np.zeros_like(image)
	if lines is not None:
		for x1,y1,x2,y2 in lines:
			# x1,y1,x2,y2 = line.reshape(-1)
			cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
	return line_image

def process_image(lane_image):
	canny_image = canny(lane_image)
	cropped_image  = region_of_interest(canny_image)
	lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
	average_lines = average_slope_intercept(lane_image,lines)
	line_image = display_line(lane_image,average_lines)
	combo_image = cv2.addWeighted(lane_image,0.5,line_image,1,1)
	cv2.imshow("result",combo_image)
	cv2.waitKey(2)

# image = cv2.imread(image_path)
# lane_image = np.copy(image)
# process_image(lane_image)
cap = cv2.VideoCapture(video_path)

while(cap.isOpened()):
	_,frame = cap.read()
	process_image(frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()



