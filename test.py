import numpy as numfun
import cv2 as cv2
import random
import os
print(numfun.__version__)
print(cv2.__version__)

cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FPS, 20)

modes = ['Search', 'Pursuit', 'Reset', 'Manual']

i = 0
wd = os.getcwd()
while(True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    label = i % 10
    cv2.imwrite(wd+ '/image/temp' + str(label) + '.jpg', frame)

    modestr = modes[random.randint(0,3)]
    altstr = str(random.randint(0,20)) + "." + str(random.randint(0,9))
    speedstr = str(random.randint(0,10)) + "." + str(random.randint(0,99))


    file = open(wd + '/text/info.txt', 'w')
    file.write(modestr + "\n")
    file.write(altstr + "\n")
    file.write(speedstr + "\n")
    file.write(str(label))

    file.close()
    i = i + 1

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
