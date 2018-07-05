#!/bin/env python

import random
import sys

global w,MisMatch
global TrainingPoints
TrainingPoints=[]
MisMatch=0
w=[0,0,0]

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

# since it's 2-d, x[0] -> x, x[1]->y
def buildTrainingData(f,N=10):
    #global TrainingPoints
    for i in xrange(N): 
        x=[random.uniform(-1,1),random.uniform(-1,1),1]
        y=f(x)
        TrainingPoints.append((x,y))

def h(x,w):
    # gn
    dotproduct=sum(value*weight for value,weight in zip(x,w))
    if dotproduct>0: return 1 
    else: return -1

def PLA():
    iter=0
    while(1):
        #print '-'*60
        iter+=1
        error_count=0
        mis=[]
        for x,y in TrainingPoints:
            #print error_count,x,y,w,h(x,w)
            if(h(x,w)!=y): # misclarified
                error_count+=1
                mis.append((x,y))
        if len(mis)==0: break
        j=random.randint(0,(len(mis)-1)) # random pick a misclassified
        x=mis[j][0]
        y=mis[j][1]
        w[0]=w[0]+y*x[0]
        w[1]=w[1]+y*x[1]
        w[2]=w[2]+y*x[2]
        if error_count==0 or iter>=10000:
            break
    #print "PLA: ",iter,w
    return iter,w

def Prob(w,f,N=1000):
    mis=0
    for i in xrange(N):
        x=[random.uniform(-1,1),random.uniform(-1,1),1]
        if(h(x,w)!=f(x)): # f!=g, here f should be unkown
            mis+=1
    return mis

if __name__=="__main__":
    iter=0
    n=1000
    TrainerNo=100
    for i in xrange(n):
        TrainingPoints=[]
        w=[0,0,0]
        tF=TargetFunc()
        buildTrainingData(tF.f,TrainerNo)
        #print TrainingPoints
        it,w=PLA() # run once
        if it!=10000: 
           iter+=it 
        else:
            n-=1
        MisMatch+=Prob(w,tF.f,n)
        del tF
    print "Avg: ",iter/n, " @n=",n
    print "Prob: ",float(MisMatch)/(n*n)
