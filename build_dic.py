import cv2
from timeit import default_timer as timer
import numpy as np
from rpca import pcp,solve_proj
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from datetime import datetime


class background:

    '''Class: Background 

    Attributes:
        BGBuildPeriod: Time elapsed during BG analysis in hours
        BuildRate: Time between BG observations in minutes
        Width: Frame Width
        Length: Frame Length
        BG_Images: BG observations
        BGDic: RPCA-based dictionary that describes the background

     Methods:
        Capture(): Capture images from background during BGBuildPeriod hours at each BuidRate minutes
        Dic(tol): Create the RPCA-based dictionary with tol tolerance '''

    def __init__(self,BGBuildPeriod=24,BuildRate=5,fastAcq=False,scalingFactor=1, prevCap=None):
        self.BGBuildPeriod=BGBuildPeriod
        self.BuildRate=BuildRate
        self.fastAcq=fastAcq
        if not self.fastAcq:
            dim=int(self.BGBuildPeriod*(60/self.BuildRate))
        else:
            dim=int(self.BGBuildPeriod/self.BuildRate)

        self.Width=int(640*scalingFactor)
        self.Length=int(480*scalingFactor)
        
        if prevCap is None:
            self.BG_Images=self.Capture(scalingFactor)
        else:
            self.BG_Images=prevCap

        self.BGDic=self.Dic(tol=1e-3)

        

    def Capture(self,scaling):
        
        cam=cv2.VideoCapture(0)
        #cam.set(2,self.Width)
        #cam.set(3,self.Length)

        

        if not self.fastAcq:
            dim=int(self.BGBuildPeriod*(60/self.BuildRate))
            frame_interval=self.BuildRate*60
        else:
            dim=int(self.BGBuildPeriod/self.BuildRate)
            frame_interval=self.BuildRate

        capture=np.zeros((self.Width*self.Length,dim))
        capturePosition=0

        start=-1e3
        while True:
            ret,frame=cam.read()
            frame=cv2.resize(frame, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA)
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame=cv2.normalize(frame, None, 1.0,0.0, cv2.NORM_MINMAX,cv2.CV_32F)
            cv2.imshow('Capture Feed',frame)
            end=timer()

            if(end-start)>frame_interval:
                frameArray=np.array(frame).reshape((self.Width*self.Length,1))
                capture[:,capturePosition]=frameArray[:,0]
                capturePosition=capturePosition+1
                if not self.fastAcq:
                    with open('BG.npy', 'wb') as f:
                        np.save(f, capture)
                now = datetime.datetime.now()
                print (str(capturePosition)+': '+now.strftime("%Y-%m-%d %H:%M:%S"))
                if(capturePosition>=dim):
                    break
                start=timer()
            
            key = cv2.waitKey(10)
            if key == 27:
                break 

        cv2.destroyAllWindows() 
        cv2.VideoCapture(0).release()
        return capture

    def Dic(self,tol=1e-3):
        #L,S,k,r=pcp(self.BG_Images,tol=tol)
        L=np.load('L_sol.npy')
        r=np.linalg.matrix_rank(L)
        self.BG_Images=None

        #U,sig,V = np.linalg.svd(L-np.mean(L),full_matrices=False)
        U,sigma,V=svds(L,k=r)
        m=self.Width*self.Length

        self.lambda1 = 1.0/np.sqrt(m)/np.mean(sigma) 
        self.lambda2 = 1.0/np.sqrt(m) # 0.05 

        with open('L.npy', 'wb') as f:
            np.save(f, L)

        with open('dic.npy', 'wb') as f:
            np.save(f, U)

        print("Rank: "+str(r))
        
        return U

    def Decompose(self,Im):
        ImArray=np.array(Im).reshape((self.Width*self.Length,1))
        alpha=np.matmul(self.BGDic.T,ImArray)
        L=np.matmul(self.BGDic,alpha)
        S=ImArray-L
        L=L.reshape((self.Length,self.Width))
        S=S.reshape((self.Length,self.Width))
        L=cv2.normalize(L, None, 1,0, cv2.NORM_MINMAX, cv2.CV_32F)
        S=cv2.normalize(S, None, 1,0, cv2.NORM_MINMAX, cv2.CV_32F)
        return L,S

    def StocDecompose(self,Im,tol=1e-7):
        m,r=self.BGDic.shape
        A = np.zeros((r, r))
        B = np.zeros((m, r))

        
        ImArray=np.array(Im).reshape((self.Width*self.Length,1))
        si,ai=solve_proj(ImArray[:,0],self.BGDic,self.lambda1,self.lambda2,tol=tol)

        A = A + np.outer(si, si)
        B = B + np.outer(ImArray[:,0] - ai, si)
        
        b_frame=np.array(self.BGDic.dot(si).reshape(self.Length,self.Width))
        a_frame=np.array(ai.reshape(self.Length,self.Width))

        S=cv2.normalize(a_frame, None, 1,0, cv2.NORM_MINMAX, cv2.CV_32F)
        L=cv2.normalize(b_frame, None, 1,0, cv2.NORM_MINMAX, cv2.CV_32F)

        return L,S
    
if __name__ == '__main__':
    scaling=0.5
    frame_interval=0.05
    plot_time=5
    bgcap=np.load('BG_24_nub.npy')
    Time_measure=True

    BG=background(scalingFactor=scaling,prevCap=bgcap)

    cam=cv2.VideoCapture(0)


    vec_size=int(plot_time/frame_interval)
    en_vec=np.zeros(vec_size)
    mov_avg=np.zeros(vec_size)
    ker_size=10
    avg_ker=np.ones(ker_size)/ker_size

    stoc_time=[]
    dic_time=[]
    t_index=0

    i=0
    start=-3
    if not Time_measure:
        plt.ion()
        fig=plt.figure()
        ax = fig.add_subplot(111)
        line1, = ax.plot(en_vec[ker_size-1:], 'b-')
        line2, = ax.plot(mov_avg[ker_size-1:], 'r-')
        plt.ylim([100, 200])
        plt.title('Anomaly energy')
    while True:
        ret,frame=cam.read()
        end=timer()
        

        if(end-start)>frame_interval:
            frame=cv2.resize(frame, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA)
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame=cv2.normalize(frame, None, 1.0,0.0, cv2.NORM_MINMAX,cv2.CV_32F)
            fp=frame
            
            t1=timer()
            L,S=BG.Decompose(frame)
            t2=timer()
            LStoc,SStoc=BG.StocDecompose(frame)
            t3=timer()
            if(Time_measure):
                dic_time.append(t2-t1)
                stoc_time.append(t3-t2)
                t_index=t_index+1
            vis1 = np.concatenate((frame,L,S), axis=1)
            vis2= np.concatenate((frame,LStoc,SStoc), axis=1)
            vis = np.concatenate((vis1,vis2), axis=0)

            cv2.imshow('frames', vis)
            fr1=timer()
            #print('New frame interval: '+str((fr1-start)))
            energy=np.linalg.norm(S)

            if not Time_measure:
                if(i<vec_size):
                    en_vec[i]=energy
                    i=i+1
                else:
                    en_vec[0:vec_size-1]=en_vec[1:vec_size]
                    en_vec[vec_size-1]=energy
                    mov_avg=np.convolve(en_vec,avg_ker,mode='valid')
                    line1.set_ydata(en_vec[ker_size-1:])
                    line2.set_ydata(mov_avg)
                    fig.canvas.draw()

            #print('Anomaly energy: '+str(energy))
            print(fr1-start)

            start=timer()
        
        if(Time_measure and t_index==100):
            print('Dic AVG: '+str(np.average(dic_time).round(decimals=4)))
            print('Dic SD: '+str(np.std(dic_time).round(decimals=4)))
            print('Stoc AVG: '+str(np.average(stoc_time).round(decimals=4)))
            print('Stoc SD: '+str(np.std(stoc_time).round(decimals=4)))
            break


        key = cv2.waitKey(10)
        if key == 27:
            break
        if key==70 or key==102:
            date = datetime.now().strftime("%Y_%m_%d-%I%M%S_%p")
            fname='./frame_'+date+'.jpg'
            fp=cv2.normalize(fp, None, 255.0,0.0, cv2.NORM_MINMAX,cv2.CV_32F)
            cv2.imwrite(fname,fp)
            fname='./S_'+date+'.jpg'
            S=cv2.normalize(S, None, 255.0,0.0, cv2.NORM_MINMAX,cv2.CV_32F)
            cv2.imwrite(fname,S)
        if key==71 or key==103:
            fname='./energy_'+date+'.png'
            plt.savefig(fname)
        if key==83 or key==115:
            date = datetime.now().strftime("%Y_%m_%d-%I%M%S_%p")
            fname='./frame_'+date+'.jpg'
            fp=cv2.normalize(fp, None, 255.0,0.0, cv2.NORM_MINMAX,cv2.CV_32F)
            cv2.imwrite(fname,fp)
            fname='./dic_'+date+'.jpg'
            S=cv2.normalize(S, None, 255.0,0.0, cv2.NORM_MINMAX,cv2.CV_32F)
            cv2.imwrite(fname,S)
            fname='./stoc_'+date+'.jpg'
            SStoc=cv2.normalize(SStoc, None, 255.0,0.0, cv2.NORM_MINMAX,cv2.CV_32F)
            cv2.imwrite(fname,SStoc)


    cv2.destroyAllWindows() 
    cv2.VideoCapture(0).release()
    