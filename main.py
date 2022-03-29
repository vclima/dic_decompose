from threading import Thread
import cv2
import time
import numpy as np
import scipy.sparse.linalg 
import rpca

class VideoStreamWidget(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        
        # decompose pars
        self.lambda1=None
        self.lambda2=None
        self.mu=None
        self.tol=1e-7  
        
        # frame setting
        self.shape=None
        self.m1,self.m2=None,None
        self.a_frame=None
        self.b_frame=None
        self.frame=None
        
        # burnin buffer
        self.burnin=100
        #self.n_components=self.burnin-1
        self.buffer=[]
        self.burned=False
        
        # basis update
        self.L=None
        self.A=None
        self.B=None
        self.bu_count=0
        self.bu_limit=np.inf
      
       

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, frame) = self.capture.read()
                frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
                frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.frame=cv2.normalize(frame, None, 1.0,0.0, cv2.NORM_MINMAX,cv2.CV_32F)
                #self.frame=self.frame-np.mean(self.frame)
                
                # update frame setting
                self.m1,self.m2=self.frame.shape
                self.shape=self.frame.shape
                
                # burn state
                if self.burned is False:
                    # update burnin buffer
                    self.buffer.append(self.frame.reshape(self.m1*self.m2))
                    if len(self.buffer)>self.burnin:
                        self.buffer.pop(0)    
                else:
                    # update decomposition
                    self.decompose_frame()
                
                    # update L matrix 
                    if self.bu_count>self.bu_limit:
                        self.update_basis()
                        self.bu_count=0
                    else: 
                        self.bu_count=self.bu_count+1

    def show_frame(self):
        # Display frames in main program
        if self.b_frame is not None and self.a_frame is not None:
#             print('y:',len(self.frame.shape))
#             print('b:',len(self.b_frame.shape))
#             print('a:',len(self.a_frame.shape))

            self.b_frame=cv2.normalize(self.b_frame, None, 1,0, cv2.NORM_MINMAX, cv2.CV_32F)
            self.a_frame=cv2.normalize(self.a_frame, None, 1,0, cv2.NORM_MINMAX, cv2.CV_32F)
            self.frame = cv2.normalize(self.frame, None, 1,0, cv2.NORM_MINMAX, cv2.CV_32F)
            
            vis = np.concatenate((self.frame,self.b_frame,self.a_frame), axis=1)
            cv2.imshow('frames', vis)
        else:
            self.frame = cv2.normalize(self.frame, None, 1,0, cv2.NORM_MINMAX, cv2.CV_32F)
            cv2.imshow('frames', self.frame)
            #print('oi')
            
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)
            
    def burn_frames(self):
        # store burnin frames and run pcp
        print('burning ',self.burnin,' first samples')
        
        buffer_arr=np.array(self.buffer).transpose()
        print(len(self.buffer))
        m, n = buffer_arr.shape
        # calculate pcp on burnin samples and find rank r
        Bhat,Ahat,niter,r=rpca.pcp(buffer_arr,tol=1e-5)
        #L,sigmas,V = np.linalg.svd(Bhat)
        print(min(Bhat.shape))
        L,sigmas,V=scipy.sparse.linalg.svds(Bhat,k=r)
        
        #
        #L=L[:,:r].dot(np.sqrt(np.diag(sigmas[:r])))
        #
        #Uhat, sigmas_hat, Vhat = scipy.sparse.linalg.svds(L,k=r)
        #U = L[:,:r].dot(np.sqrt(np.diag(sigmas[:r])))
        #
        #print('l.shape',L.shape)
        #print('u.shape',U.shape)
        
        self.L=L        
        
        
        print('rank=',r)
        
        # pars
        self.burned=True
        self.lambda1 = 1.0/np.sqrt(m)/np.mean(sigmas) 
        self.lambda2 = 1.0/np.sqrt(m) # 0.05 
        
        print('lambda1=',self.lambda1)
        print('lambda2=',self.lambda2)
        
        # aux vars
        self.A = np.zeros((r, r))
        self.B = np.zeros((m, r))
    
    
    def decompose_frame(self):
        si,ai=rpca.solve_proj(self.frame.reshape(self.m1*self.m2),self.L,self.lambda1,self.lambda2,self.mu,tol=self.tol)

        self.A = self.A + np.outer(si, si)
        self.B = self.B + np.outer(self.frame.reshape(self.m1*self.m2) - ai, si)
        
        self.b_frame=np.array(self.L.dot(si).reshape(self.shape))
        self.a_frame=np.array(ai.reshape(self.shape))
        
    def update_dicio(self):
        self.L=rpca.update_dicio(self.L, self.A, self.B, self.lambda1)
        
if __name__ == '__main__':
    # defaults
    video_src=0
    sleep_in=5
    sleep_out=1
    scaling_factor=0.5
    shape=(240, 320)
    m1,m2=shape

    # start widget  
    time.sleep(sleep_in) 
    video_stream_widget = VideoStreamWidget(src=video_src)
    time.sleep(sleep_out)
    
    # pre-burn stage
    while len(video_stream_widget.buffer)<video_stream_widget.burnin:
        try:
            video_stream_widget.show_frame()
            print(len(video_stream_widget.buffer))
        except AttributeError:
            pass
        
    # burn stage
    print('burning frames...')
    video_stream_widget.burn_frames()
    
    # post-burn stage
    while True:
        try:
            video_stream_widget.show_frame()
        except AttributeError:
            pass
