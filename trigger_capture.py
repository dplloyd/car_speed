# Python program to implement  
# Webcam Motion Detector 
  
# importing OpenCV, time and Pandas library 
import cv2, time, pandas, argparse 
# importing datetime class from datetime library 
from datetime import datetime
from imutils import video

# https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', help='path to the video file')
ap.add_argument('-a', '--min-area', type=int, default=500,
                help='minimum area size')
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam

if args.get('video', None) is None:
    vc = video.VideoStream(src=0).start()
    time.sleep(2.0)
else:

     # otherwise, we are reading from a video file
    vc = cv2.VideoCapture(args['video'])
    
    
#Image resize function for displaying on laptop
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

  
# Assigning our static_back to None 
static_back = None

# Assigning our previous frame to None
gray_prev = None
  
# List when any moving object appear 
motion_list = [ None, None ] 
  
# Time of movement 
time = [] 
  
# Initializing DataFrame, one column is start  
# time and other column is end time 
df = pandas.DataFrame(columns = ["Start", "End"])


# Capturing video
# Use this if just running interactively and want to read from webcam
#vc = cv2.VideoCapture(1)

#Take a background photo
print("Capture background - press space when ready")

while True: 
    # Reading frame(image) from video 
    check, frame = vc.read() 
    key = cv2.waitKey(1)
    cv2.imshow("Color Frame", frame) 

    # if q entered whole process will stop 
    if key%256 == 32:
        bkgrd_image = frame
        # SPACE pressed
        print("Space hit, closing...")
        break

cv2.destroyAllWindows()


bkgrd_image = cv2.cvtColor(bkgrd_image, cv2.COLOR_RGB2GRAY)
resize_bkgrd_image = ResizeWithAspectRatio(bkgrd_image, width=400) # Resize by width OR
cv2.imshow("bkgrd", resize_bkgrd_image)

iteration = 0 
# Infinite while loop to treat stack of image as video 
while True: 
    # Reading frame(image) from video 
    check, frame = vc.read() 

    # Initializing motion = 0(no motion) 
    motion = 0

    # detect if this is the first iteration
    iteration = iteration + 1


  
    # Converting color image to gray_scale image 
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


    # Converting gray scale image to GaussianBlur  
    # so that change can be find easily 
    gray = cv2.GaussianBlur(gray, (71, 71), 0)
    
    # Similarly, apply blur to background if the first iteration, otherwise use previous frame
    if iteration>1:
        static_back = gray_prev
    else:
        static_back = cv2.GaussianBlur(bkgrd_image,(71,71),0)

  
    # Difference between static background  
    # and current frame(which is GaussianBlur) 
    diff_frame = cv2.absdiff(static_back, gray) 
  
    # If change in between static background and 
    # current frame is greater than 30 it will show white color(255) 
    thresh_frame = cv2.threshold(diff_frame, 10, 255, cv2.THRESH_BINARY)[1] 
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2) 
  
    # Finding contour of moving object 
    cnts,_ = cv2.findContours(thresh_frame.copy(),  
                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
  
    for contour in cnts: 
        if cv2.contourArea(contour) < 10000: 
            continue
        motion = 1
  
        (x, y, w, h) = cv2.boundingRect(contour) 
        # making green rectangle arround the moving object 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3) 
  
    # Appending status of motion 
    motion_list.append(motion) 
  
    motion_list = motion_list[-2:] 
  
    # Appending Start time of motion 
    if motion_list[-1] == 1 and motion_list[-2] == 0: 
        time.append(datetime.now()) 
  
    # Appending End time of motion 
    if motion_list[-1] == 0 and motion_list[-2] == 1: 
        time.append(datetime.now()) 
  
    # Displaying image in gray_scale
    resize_gray = ResizeWithAspectRatio(gray, width=400) # Resize by width OR
    cv2.imshow("Gray Frame", resize_gray) 
  
    # Displaying the difference in currentframe to 
    # the staticframe(very first_frame)
    resize_diff_frame = ResizeWithAspectRatio(diff_frame, width=400) # Resize by width OR
    cv2.imshow("Difference Frame", resize_diff_frame) 
  
    # Displaying the black and white image in which if 
    # intensity difference greater than 30 it will appear white
    resize_thresh_frame = ResizeWithAspectRatio(thresh_frame, width=400) # Resize by width OR
    cv2.imshow("Threshold Frame", resize_thresh_frame) 
  
    # Displaying color frame with contour of motion of object
    resize_frame = ResizeWithAspectRatio(frame, width=400) # Resize by width OR
    cv2.imshow("Color Frame", resize_frame) 
  
    key = cv2.waitKey(1) 
    # if q entered whole process will stop 
    if key == ord('q'): 
        # if something is movingthen it append the end time of movement 
        if motion == 1: 
            time.append(datetime.now()) 
        break

    # Assign current frame to a new frame to act as background on next iteration
    gray_prev = gray
  
# Appending time of motion in DataFrame 
for i in range(0, len(time), 2): 
    df = df.append({"Start":time[i], "End":time[i + 1]}, ignore_index = True) 
  
# Creating a CSV file in which time of movements will be saved 
df.to_csv("Time_of_movements.csv") 
  
vc.release() 
  
# Destroying all the windows 
cv2.destroyAllWindows() 
