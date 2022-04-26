import cv2
import numpy as np
from timeit import default_timer as timer
import imageio


def show_image(img,t):
    s,l=img.shape
    img_save=[]
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
        img_save.append(cv2.normalize(frame, None, 255,0.0, cv2.NORM_MINMAX,cv2.CV_32F))
        #img_save.append(frame)

    imageio.mimsave('BG_24_sol.gif', img_save,fps=5)
        



bgcap=np.load('BG_24_sol.npy')
dic=np.load('dic.npy')
L=np.load('L_sol.npy')

#show_image(np.flip(dic,axis=1),0.5)
#show_image(bgcap,0.1)



img_save=[]
s,l=bgcap.shape
for i in range(l):
    frame=bgcap[:,i]
    frame=frame.reshape(240,320)
    frame=cv2.normalize(frame, None, 1.0,0.0, cv2.NORM_MINMAX,cv2.CV_32F)

    frame2=L[:,i]
    frame2=frame2.reshape(240,320)
    frame2=cv2.normalize(frame2, None, 1.0,0.0, cv2.NORM_MINMAX,cv2.CV_32F)
    
    #fr=np.concatenate((frame,frame2),axis=1)
    cv2.imshow('Capture Feed',frame2)
    key = cv2.waitKey(10)
    if key == 27:
        break
    now=timer()
    while((timer()-now)<0.05):
        pass
    
    img_save.append(cv2.normalize(frame2, None, 255,0.0, cv2.NORM_MINMAX,cv2.CV_32F))

imageio.mimsave('L_sol2.gif', img_save,fps=20)
