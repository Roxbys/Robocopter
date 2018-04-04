import cv2
import numpy as np
import os
def foobar(x):
    return
cap = cv2.VideoCapture(0)
cv2.namedWindow('image')
'''cv2.createTrackbar('H_min','image', 0,255, foobar)
cv2.createTrackbar('H_max','image', 255,255, foobar)
cv2.createTrackbar('S_min','image', 0,255, foobar)
cv2.createTrackbar('S_max','image', 255,255, foobar)
cv2.createTrackbar('V_min','image', 0,255, foobar)
cv2.createTrackbar('V_max','image', 255,255, foobar)
'''
lower_range = np.array([0,0,0])
upper_range = np.array([255,255,255])

kernel = np.ones((5,5), np.uint8)
i = 0
wd = os.getcwd()
while True:
    ret, img = cap.read()
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    '''h_min = cv2.getTrackbarPos('H_min','image')
    h_max = cv2.getTrackbarPos('H_max','image')
    s_min = cv2.getTrackbarPos('S_min','image')
    s_max = cv2.getTrackbarPos('S_max','image')
    v_min = cv2.getTrackbarPos('V_min','image')
    v_max = cv2.getTrackbarPos('V_max','image')'''


    prev_lower = lower_range
    prev_upper = upper_range
    '''lower_range = np.array([h_min,s_min,v_min])
    upper_range = np.array([h_max,s_max,v_max])'''

    lower_range = np.array([94,172,103])
    upper_range = np.array([146,255,170])
    if (np.any(prev_lower != lower_range) or np.any(prev_upper != upper_range)):
        print(str(lower_range) + " " + str(upper_range))
    mask = cv2.inRange(hsv_img, lower_range, upper_range)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #erode = cv2.erode(mask,kernel, iterations = 1)
    #dilate = cv2.dilate(img,kernel,iternations = 1)
    _,contours, _= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if(len(contours) < 1):
        print("srry no contours")
    else:
        cv2.drawContours(img, contours, -1, (0,255,0), 2)
        cont = contours[0]
        #moment = cv2.moments(cont)
        x,y,w,h = cv2.boundingRect(cont)
        #cv2.rectangle(img,(x,y), (x+w, y+h), (0,255,0),2)
        cv2.putText(img,'x: '+str(x)+',y: '+str(y),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
    label = i % 10
    cv2.imwrite(wd+ '/web/image/temp' + str(label) + '.jpg', img)
    i = i + 1
        #cx = int(moment['m10']/moment['m00'])
        #cy = int(moment['m01']/moment['m00'])
        #res = cv2.bitwise_and(img, img, mask)
        #cv2.imshow('hsv', hsv_img)
        #cv2.imshow('mask', mask)
    #cv2.imshow('opening', opening)
    #cv2.imshow('img',img)
    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
