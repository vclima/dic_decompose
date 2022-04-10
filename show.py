import cv2
import numpy as np
from timeit import default_timer as timer


def show_image(img,t):
    s,l=img.shape
    for i in range(l):
        frame=img[:,i]
        frame=frame.reshape(240,320)
        frame=cv2.normalize(frame, None, 1.0,0.0, cv2.NORM_MINMAX,cv2.CV_32F)
        cv2.imshow('Capture Feed',frame)
        key = cv2.waitKey(10)
        if key == 27:
            return
        now=timer()
        while((timer()-now)<t):
            pass



bgcap=np.load('BG_24_sol.npy')
dic=np.load('dic.npy')
L=np.load('L_sol.npy')

show_image(np.flip(dic,axis=1),0.5)
#show_image(bgcap,0.1)

'''
s,l=bgcap.shape
for i in range(l):
    frame=bgcap[:,i]
    frame=frame.reshape(240,320)
    frame=cv2.normalize(frame, None, 1.0,0.0, cv2.NORM_MINMAX,cv2.CV_32F)

    frame2=L[:,i]
    frame2=frame2.reshape(240,320)
    frame2=cv2.normalize(frame2, None, 1.0,0.0, cv2.NORM_MINMAX,cv2.CV_32F)
    
    fr=np.concatenate((frame,frame2),axis=1)
    cv2.imshow('Capture Feed',fr)
    key = cv2.waitKey(10)
    if key == 27:
        break
    now=timer()
    while((timer()-now)<0.1):
        pass
'''