import numpy as np
import argparse
#import imutils
import cv2

#USAGE
#python color_segment3.py -i garbagebags2.jpg -o output.png -c rgb/hsv


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="colouredimage.jpg",
    help="path to the input image")
ap.add_argument("-c", "--color", default="hsv",
    help="define the color space you want to use for segmentation, RGB,HSV")
args = vars(ap.parse_args())
  
# dict to count
counter = {}
 
# load the image
image_orig = cv2.imread(args["image"])
height_orig, width_orig = image_orig.shape[:2]
Color=args['color']

if Color=='hsv':
    cv2.imshow("input",image_orig)
    image_orig=cv2.cvtColor(image_orig,cv2.COLOR_BGR2HSV)

# output image with contours
image_contours = image_orig.copy()
 
# colors to detect

colors = ['blue','yellow','red','green']

for color in colors:
 
    # copy of original image
    image_to_process = image_orig.copy()
 
    # initializes counter
    counter[color] = 0
 
    # define NumPy arrays of color boundaries (BGR/HSV vectors)
    if color == 'blue' and Color=='hsv':
        lower = np.array([101,50,38])
        upper = np.array([110,255,255])
        
    elif color=='yellow' and Color=='hsv':
        lower=np.array([20,100,100]) #110,238,219
        upper= np.array([30,255,255]) #130,238,219

    elif color=='red' and Color=='hsv':
        lower = np.array([160,20,70]) #[30,150,50]
        upper = np.array([190,255,255])#[255,255,180]

    elif color == 'green' and Color == 'hsv':
        lower=np.array([36,25,25])
        upper=np.array([70,255,255])

         
    
    # find the colors within the specified boundaries
    image_mask = cv2.inRange(image_to_process, lower, upper)
    image_res = cv2.bitwise_and(image_to_process, image_to_process, mask = image_mask)
    
    # PREVIEW THE MASKED IMAGE
    while True:
        cv2.imshow(color,image_res)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    a,c,image_gray = cv2.split(image_res)
    image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
    
    # MORPHOLOGICAL TRANSFORMATION
    # perform edge detection, then perform a dilation + erosion to close gaps in between object edges
    image_edged = cv2.Canny(image_gray, 50, 100)
    
    image_edged = cv2.dilate(image_edged, None, iterations=10)
    
    # it erodes away the boundaries of foreground object
    image_edged = cv2.erode(image_edged, None, iterations=10) 
 
    # find contours in the edge map
    cnts = cv2.findContours(image_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    #The cv2.findContours function in OpenCV 2.4 returns a 2-tuple while in OpenCV 3 it returns a 3-tuple

    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # loop over the contours individually
    for c in cnts:
         
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 30:
            continue
         
        # compute the Convex Hull of the contour

        hull = cv2.convexHull(c)
        
        if color == 'blue':
            # prints contours in rgb blue color
            cv2.drawContours(image_contours,[hull],0,(0,0,255),1) 
          
        elif color == 'yellow':
            # prints contours in rgb red  color
            cv2.drawContours(image_contours,[hull],0,(255,0,0),1)
            
        elif color == 'red':
            # prints contours in rgb  green color
            cv2.drawContours(image_contours,[hull],0,(60, 100, 50),1) # Yellow 255,255,0
            
        elif color == 'green':
            # prints contours in rgb yellow color
            cv2.drawContours(image_contours,[hull],0,(255, 128, 0),1)
        counter[color] += 1

    print("{} {} Counts".format(counter[color],color))

if Color=='hsv':
    image_contours=cv2.cvtColor(image_contours,cv2.COLOR_HSV2BGR)

else:
    pass

