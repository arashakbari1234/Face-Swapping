import cv2
import numpy as np
import json

# Create point matrix get coordinates of mouse click on image
point_matrix = np.zeros((6,2),np.int)
counter = 0 

def mousePoints(event, x, y, flags, params):
    global counter 
    # Left button mouse click event opencv
    if event == cv2.EVENT_LBUTTONDOWN:
        point_matrix[counter] = int(x), int(y)
        counter += 1


# Read image
image1 = cv2.imread('./images/Im386.bmp')
image1 = cv2.resize(image1, (400, 500))

# image1 = cv2.imread('./images/Im387.bmp')
# image1 = cv2.resize(image1, (400, 500))
# cv2.imwrite('image2_resized.jpg', image1)

while True:
    # show circle of points
    num_points = 6
    for x in range (0,num_points):
        cv2.circle(image1,(point_matrix[x][0],point_matrix[x][1]),3,(0,255,0),cv2.FILLED)

    if counter == num_points:
        # point_matrix = [[item[0], item[1]] for item in point_matrix]
        # save in json
        with open("Im387.json", "w") as f:
            json.dump(point_matrix.tolist(), f)
        print(point_matrix)
        cv2.waitKey(1000)
        break


    # Showing original image
    cv2.imshow("Original Image ", image1)
    cv2.setMouseCallback("Original Image ", mousePoints)
    

    if cv2.waitKey(20) & 0xff == ord('q'): break
# cv2.waitkey(5000)
cv2.destroyAllWindows()







