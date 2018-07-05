'''
Created on Apr 7, 2013
 
@author: Silvrous
'''
# Perceptron Learning Algorithm
import random
 
 
 
def sign (k):           # Sign function
    if k>=0:
        return 1
    else:
        return -1
   
def cp(a,b):                                    #Inner product of two vectors
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
 
nr = 0
nr_diff = 0
nr_points = 100
nr_test_points = 1000
nr_iterations = 1000
 
for iterator in range(nr_iterations):      #Repeating the algorithm for the required number of iterations
    print iterator
    points= []                                #Points
    tru= []                               #True sign of each point
 
    xa=random.uniform(-1,1)             #Generating the separating line by creating two random points
    xb=random.uniform(-1,1)
    ya=random.uniform(-1,1)
    yb=random.uniform(-1,1)
 
    #line: ax + by + c = 0
    w1=1/(xb-xa)                    # a              # the three parameters can be found by using the equation (x-xa)/(xb-xa)=(y-ya)/(yb-ya)
    w2=-1/(yb-ya)                   # b
    w0= ya/(yb-ya) - xa/(xb-xa)     # c
 
    line=[w0,w1,w2]
 
    for i in range(nr_points):                   #Generating the training set of points, and saving the real sign of the points in the "tru" vector by computing inner product
        x=random.uniform(-1,1)
        y=random.uniform(-1,1)
        points.append([1,x,y])
        tru.append(sign(cp(line,[1,x,y])))
       
    rdy = False
 
    hyp=[0,0,0]               #Hypothesis begins with weights of 0
 
 
    while not rdy:              #While there are still misclassified points
        misclass=[]              #Set of misclassified points
        truclass=[]             #Real signs of the misclassified points
        nr+=1                    # Number of iterations for the homework answer
        rdy = True
        for i in range(len(points)):
            if sign(cp(hyp,points[i])) != tru[i]:      #If the point is misclassified
                misclass.append(points[i])             #Add it to the list of misclassified points
                truclass.append(tru[i])
                rdy=False
       
        if not rdy:                                       #If theere are misclassified points, choose a random misclassified point to rectify
            q=random.randint(0,len(misclass)-1)                  
            hyp[0]+= truclass[q]*misclass[q][0]             # w = w+ y*x
            hyp[1]+= truclass[q]*misclass[q][1]
            hyp[2]+= truclass[q]*misclass[q][2]
           
   
    for i in range(nr_test_points):      #Generating many points to test the hypothesis to find P(g!=f)
        x=random.uniform(-1,1)
        y=random.uniform(-1,1)
        lpoints=[1,x,y]
        if sign(cp(hyp,lpoints)) != sign(cp(line,lpoints)):           #If the points differ in their classification
            nr_diff += 1                                              
 
       
print nr*1.0/nr_iterations            #Average number of iterations until convergence, multiplied by 1.0 to get a floating-point value
print nr_diff * 1.0 / (nr_iterations * nr_test_points)  #Same, for the probability
