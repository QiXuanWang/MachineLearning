#!/bin/env python

import random
import sys
import numpy as np
import numpy.linalg as linalg

global w
global TrainingPoints
TrainingPoints=[]

# target function f
class TargetFunc(object):
    def __init__(self):
        self.point0=(random.uniform(-1,1) ,random.uniform(-1,1))
        self.point1=(random.uniform(-1,1) ,random.uniform(-1,1))

    def f(self,x):
        point0=self.point0
        point1=self.point1
        x1=((point1[1]-point0[1])/(point1[0]-point0[0]))*x[0]+(point1[0]*point0[1]-point0[0]*point1[1])/(point1[0]-point0[0])
        #print x1,x[1]
        if (x[1]>x1): return 1
        else: return -1

def h(x,w):
    # gn, or: W^T*Xn
    #dotproduct=sum(value*weight for value,weight in zip(x,w))
    dotproduct=np.dot(w.T,x)
    #print dotproduct
    if dotproduct>0: return 1 
    else: return -1

def PLA(mX,mY,w):
    iter=0
    while(1):
        #print '-'*60
        iter+=1
        error_count=0
        mis=[]
        for i in xrange(len(mY)):
            if(h(mX2[i],w)!=mY2[i]):
                error_count+=1
                mis.append((mX2[i],mY2[i]))
        if len(mis)==0: break
        j=random.randint(0,(len(mis)-1)) # random pick a misclassified
        x=mis[j][0]
        y=mis[j][1]
        w+=y*x
        #w[0]=w[0]+y*x[0]
        #w[1]=w[1]+y*x[1]
        #w[2]=w[2]+y*x[2]
        if error_count==0 or iter>=10000:
            break
    #print "PLA: ",iter,w
    return w,iter

def LRA(mX,mY):
    #print mX.shape, mY.shape
    Ein=0
    w=np.dot(linalg.inv(np.dot(mX.T,mX)),np.dot(mX.T,mY))
    return w,linalg.norm(np.dot(mX,w)-mY)/len(mY)

class Experiment(object):
    def __init__(self,f,TrainerNo=10):
        self.TrainingPoints=[]
        self.TrainerNo=TrainerNo
        self.TargetFunc=f

    # since it's 2-d, x[0] -> x, x[1]->y
    def buildTrainingData(self):
        #global TrainingPoints
        for i in xrange(self.TrainerNo): 
            x=[random.uniform(-1,1),random.uniform(-1,1),1]
            y=self.TargetFunc(x)
            self.TrainingPoints.append((x,y))

    def buildMatrix(self):
        a=[]
        b=[]
        for x,y in self.TrainingPoints:
            a.append(x)
            b.append(y)
        mX=np.array(a)
        mY=np.array(b)
        return mX,mY


if __name__=="__main__":
    ExpNo=1000
    TrainerNo=100
    TrainerNo2=1000
    Ein=0
    Eout=0
    iter=0
    for i in xrange(ExpNo):
        tF=TargetFunc()
        myExp=Experiment(tF.f,TrainerNo)
        myExp.buildTrainingData()
        mX,mY=myExp.buildMatrix()
        #print mX,mY
        myExp2=Experiment(tF.f,TrainerNo2)
        myExp2.buildTrainingData()
        mX2,mY2=myExp2.buildMatrix()

        w,E=LRA(mX,mY) # run once
        wPLA,it=PLA(mX,mY,w)
        Ein+=E
        iter+=it
        for i in xrange(len(mY2)):
            #print h(mX2[i],w),mY2[i]
            if(h(mX2[i],w)!=mY2[i]): Eout+=1
        del tF
    print "Ein: ",Ein/ExpNo
    print "Eout: ",float(Eout)/TrainerNo2/ExpNo
    print "PLA Iter: ",iter/ExpNo
