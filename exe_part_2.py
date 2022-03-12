from re import I
import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from os.path import join
import cv2


def triangle(landmark, imshow=True):
    # this function draw landmarks and return triangle
    rect = [0,0,600,600]
    subdiv = cv2.Subdiv2D(rect)
    
    subdiv.insert(landmark['point'])
    
    triangles = subdiv.getTriangleList()
    # print(triangles)
    return triangles


def warp(image_0, json_0, image_1, json_1):
    img0 = cv2.imread(image_0)
    img1 = cv2.imread(image_1)

    img1_new_face = np.zeros_like(img1)

    f1 = open(json_0)
    landmark_0 = json.load(f1)
    tri_0 = triangle(landmark_0)
    
    f2 = open(json_1)
    landmark_1 = json.load(f2)

    tri_1 = []

    
    for triangles in tri_0:
        tmp = []
        points = landmark_1['point']
        x1, y1, x2, y2, x3, y3 = triangles[0], triangles[1], triangles[2], triangles[3], triangles[4], triangles[5]
        index1 = landmark_0['point'].index([x1, y1])
        tmp.append(points[index1][0])
        tmp.append(points[index1][1])
        index2 = landmark_0['point'].index([x2, y2])
        tmp.append(points[index2][0])
        tmp.append(points[index2][1])
        index3 = landmark_0['point'].index([x3, y3])
        tmp.append(points[index3][0])
        tmp.append(points[index3][1])

        tri_1.append(tmp)
    
    for triangles,triangles2 in zip(tri_0, tri_1):
        x1, y1, x2, y2, x3, y3 = triangles[0], triangles[1], triangles[2], triangles[3], triangles[4], triangles[5]
        print(x1, y1, x2, y2, x3, y3)
        # cv2.line(img0, (int(x1),int(y1)), (int(x2), int(y2)), (0,0,255), 2)
        # cv2.line(img0, (int(x1),int(y1)), (int(x3), int(y3)), (0,0,255), 2)
        # cv2.line(img0, (int(x3),int(y3)), (int(x2), int(y2)), (0,0,255), 2)
        t1 = np.array([ [x1,y1],[x2,y2],[x3,y3] ], np.int32)
        rect1 = cv2.boundingRect(t1)
        (x, y, w, h) = rect1
        cropped_triangle = img0[y : y + h, x : x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)
        points = np.array([[x1-x, y1-y], [x2 - x, y2 - y], [x3 - x, y3- y]], np.int32)
        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
        cropped_triangle = cv2.bitwise_or(cropped_triangle, cropped_triangle, mask=cropped_tr1_mask)
        
        
    
        x1, y1, x2, y2, x3, y3 = triangles2[0], triangles2[1], triangles2[2], triangles2[3], triangles2[4], triangles2[5]
        t2 = np.array([ [x1,y1],[x2,y2],[x3,y3] ], np.int32)
        rect2 = cv2.boundingRect(t2)
        (x, y, w, h) = rect2
        cropped_triangle2 = img1[y : y + h, x : x + w]
        cropped_tr1_mask2 = np.zeros((h, w), np.uint8)
        points2 = np.array([[x1-x, y1-y], [x2 - x, y2 - y], [x3 - x, y3- y]], np.int32)
        cv2.fillConvexPoly(cropped_tr1_mask2, points2, 255)
        cropped_triangle2 = cv2.bitwise_or(cropped_triangle2, cropped_triangle2, mask=cropped_tr1_mask2)
        
        # cv2.line(img1, (x1,y1), (x2, y2), (0,0,255), 2)
        # cv2.line(img1, (x3,y3), (x2, y2), (0,0,255), 2)
        # cv2.line(img1, (x1,y1), (x3, y3), (0,0,255), 2)

        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))   
        
        triangle_area = img1_new_face[y:y+h, x:x+w] 
        triangle_area = cv2.add(triangle_area, warped_triangle)
        img1_new_face[y:y+h, x:x+w] = triangle_area

# face swapped
    img1_new_face_gray = cv2.cvtColor(img1_new_face, cv2.COLOR_BGR2GRAY)
    _, background = cv2.threshold(img1_new_face_gray, 1, 255, cv2.THRESH_BINARY_INV)
    background = cv2.bitwise_and(img1, img1, mask=background)

    result = cv2.add(background, img1_new_face)
    
    

    cv2.imshow('fdfd', result)
    cv2.imshow("Image 1", img0)
    cv2.imshow("image2", img1)
    # cv2.imshow("cropped triangle 1", background)
    # cv2.imshow("cropped triangle 2", cropped_triangle2)
    # cv2.imshow("Warped triangle", warped_triangle)  
    cv2.waitKey()
    


    #  ... = creat_mask(triangles_0.)
    # .....
    # affine transform
    # ....

    
    # cv2.imshow('test', output)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    pass


def creat_mask(triangles, image):
    # (x, y, w, h) = rec_triangles
    # crop_image = image
    # point =  # new point in rec
    # ...
    #
    # roi = cv2.bitwise_...
    pass


if __name__ == '__main__':
    json_0 = './images/Im387.json'
    json_1 = './images/Im386.json'
    image_0 = './images/Im387_resize_.png'
    image_1 = './images/Im386_resize_.png'
    warp(image_0, json_0, image_1, json_1)
    # triangle()

