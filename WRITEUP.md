# Advanced Lane Lines

## Camera Calibration
The following steps are followwed:
1. Prepare object points and image points
2. Grayscale the image
3. Find the chess corners

Store the cameraMatrix and distance coefficient using pickle in calibration.p

```
for idx, image in enumerate(images):
    
    img = mpimg.imread(image)
    # convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # 9X6 chessboard
    nx=9
    ny=6
    # get corners in the chessboard
    ret, corners = cv2.findChessboardCorners(gray,(nx,ny),None)
    
    if ret == True:
        # append object points 
        objpoints.append(objp)
        # append image points
        imgpoints.append(corners)

image = cv2.imread('./camera_cal/calibration1.jpg')
image_size = (image.shape[1],image.shape[0])

# Get Camera matrix and distortion coefficent
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints,image_size,None,None)
```

## Gradient and color spaces
It is noticed that S channel in HLS color space indicates lanes relatively better than grayscale. So we will be applying gradients operation on S channel.
Combining the following threshold magnitude, direction of the gradient to get binary image
```
    S = image_hls[:,:,2]
    # Sobel X
    gradx = abs_sobel_thresh(S,(10,150))
    # Sobel y
    grady = abs_sobel_thresh(S,(10,150),'y')
    # Magnitude
    mag = mag_thresh(S,(10,150))
    # Direction
    direction = dir_threshold(S,(0.7,1.2))
    combined = np.zeros_like(gradx)
    # combination
    combined[((gradx==1) & (grady==1)) | ((mag==1) & (direction==1))] =1
```
## Perspective Transform
ROI region is defined, trapezoid with four points
```
# mark a rectangle
left_bottom = (190,720)
left_top = (550,480)
right_top = (740,480)
right_bottom = (1150,720)
```
## Sliding Window
1.Loop through each window in nwindows
2.Find the boundaries of our current window. This is based on a combination of the current window's starting point (leftx_current and rightx_current), as well as the margin you set in the hyperparameters.
3.Use cv2.rectangle to draw these window boundaries onto our visualization image out_img. This is required for the quiz, but you can skip this step in practice if you don't need to visualize where the windows are.
4.Now that we know the boundaries of our window, find out which activated pixels from nonzeroy and nonzerox above actually fall into the window.
5.Append these to our lists left_lane_inds and right_lane_inds.
6.If the number of pixels you found in Step 4 are greater than your hyperparameter minpix, re-center our window (i.e. leftx_current or rightx_current) based on the mean position of these pixels.
```
def find_lanes(binary_warped):
    
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:],axis=0)
    out_img = np.dstack((binary_warped,binary_warped,binary_warped))*255
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) +midpoint
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
         # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    return leftx, lefty, rightx, righty, out_img
```

## Radius and Distance
Calculate the curvature from left and right curvature separately
```
def calculate_radius_of_curvature(binary_warped,left_fit,right_fit):
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty +left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    positionCar= binary_warped.shape[1]/2
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    
    y_eval=np.max(ploty)
    
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    
    left_lane_bottom = (left_fit[0]*y_eval)**2 + left_fit[0]*y_eval + left_fit[2]
    right_lane_bottom = (right_fit[0]*y_eval)**2 + right_fit[0]*y_eval + right_fit[2]
    
    actualPosition= (left_lane_bottom+ right_lane_bottom)/2
    
    distance= (positionCar - actualPosition)* xm_per_pix
    
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    return (left_curverad + right_curverad)/2, distance
```

## Define Pipeline
```
def pipeline(img):
    # apply combined gradient
    img = combined_gradient(img)
    # transform the image
    binary_warped = getWarped(img)
    # measure the lanes
    leftx, lefty, rightx, righty, out_img = find_lanes(binary_warped)
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # draw the line using left_fit and right_fit
    output = draw_line(img, left_fit, right_fit)
    ouput = cv2.cvtColor( output, cv2.COLOR_BGR2RGB )
    # get radius and curvature 
    radius, distance= calculate_radius_of_curvature(binary_warped,left_fit,right_fit)
    cv2.putText(output,"Radius of Curvature is "+ str(int(radius))+ "m", (100,100), 2, 1, (0,255,0),2 )
    cv2.putText(output,"Distance from center is {:2f}".format(distance)+ "m", (100,150), 2, 1, (0,255,0),2)
    return output
```

## Discussion

1. Encountered shadow, discoloration in the images needs to be considered and could be solved by tuning parameters and varying color spaces
2. The performance could be improved by searching from prior and making use of frame to frame information
3. Challenge video and harder challenge video should be looked at and needs to be improved
