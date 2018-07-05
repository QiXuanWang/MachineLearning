#!/bin/env python

import numpy.random as random
import time

class Coin(object):
    def __init__(self):
        self.states=[]
    def tossN(self,N=1):
        self.states = [random.randint(0,2) for i in xrange(N)]
        

class Experiment(object):
    def __init__(self,tossNo=10,coinNo=1000):
        self.tossNo=tossNo
        self.coinNo=coinNo
        self.Coins=[]

    def buildCoins(self):
        for i in xrange(self.coinNo):
            self.Coins.append(Coin())
        
    def run(self):
        for eachCoin in self.Coins:
            eachCoin.tossN(self.tossNo)
        V1=sum(self.Coins[0].states)
        Vrand=sum(self.Coins[random.randint(0,self.coinNo)].states)
        Vmin=min([sum(eachCoin.states) for eachCoin in self.Coins])
        return float(V1)/self.coinNo,float(Vrand)/self.coinNo,float(Vmin)/self.coinNo

if __name__ == "__main__":
    tossNo=10
    CoinNo=1000
    ExpNo=100000
    V1=0
    Vrand=0
    Vmin=0
    time.clock()
    for i in xrange(ExpNo):
        myExp=Experiment(tossNo,CoinNo)
        myExp.buildCoins()
        results=myExp.run()
        V1+=results[0]
        Vrand+=results[1]
        Vmin+=results[2]
        del myExp
    print time.clock()
    print float(V1)/ExpNo,float(Vrand)/ExpNo,float(Vmin)/ExpNo


