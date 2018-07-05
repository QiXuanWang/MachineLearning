#!/bin/env python

import random
import sys

global w,MisMatch
global TrainingPoints
TrainingPoints=[]
MisMatch=0
w=[0,0]

# target function f
class TargetFunc(object):
    def __init__(self):
        self.point0=(random.uniform(-1,1) ,random.uniform(-1,1))
        self.point1=(random.uniform(-1,1) ,random.uniform(-1,1))

    #([x0,x1],y)
    def f(self,x):
        point0=self.point0
        point1=self.point1
        x1=((point1[1]-point0[1])/(point1[0]-point0[0]))*x[0]+(point1[0]*point0[1]-point0[0]*point1[1])/(point1[0]-point0[0])
        if x[1]>x1: return 1
        else: return -1

# since it's 2-d, x[0] -> x, x[1]->y
def buildTrainingData(f,N=10):
    global TrainingPoints
    for i in xrange(N): 
        x=[random.uniform(-1,1),random.uniform(-1,1)]
        y=f(x)
        TrainingPoints.append((x,y))

def h(x,w):
    # gn
    dotproduct=sum(value*weight for value,weight in zip(x,w))
    if dotproduct>0: return 1
    else: return -1

def PLA():
    global MisMatch
    global w
    iter=0
    #for x,y in TrainingPoints:
    #    iter=0
    #    if(h(x,w)!=f(x)): # f!=g, here f should be unkown
    #        MisMatch+=1
    while(1):
        print '-'*60
        iter+=1
        error_count=0
        for x,y in TrainingPoints:
            print error_count,x,y,w,h(x,w)
            if(h(x,w)!=y): # misclarified
                error_count+=1
                w[0]=w[0]+y*x[0]
                w[1]=w[1]+y*x[1]
        if error_count==0 or iter>20:
            break
    print "PLA: ",iter,w,h(x,w),y

if __name__=="__main__":
    for i in xrange(1):
        tF=TargetFunc()
        buildTrainingData(tF.f,10)
        #print TrainingPoints
        PLA() # run once


