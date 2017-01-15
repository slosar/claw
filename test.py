#!/usr/bin/env python
from __future__ import print_function, division
import claw
import numpy as np, healpy as hp,matplotlib.pyplot as plt, cPickle as cp
from os.path import isfile


flatWeight=True
zeroPower=True
Ng=10000
Ngtest=1000
pname="./MeasureCl_%i_%i_%i.pickle"%(flatWeight,zeroPower,Ng)


def main():
    ClT=makeTestCl()
    weight,noise=makeTestWindowNoise(ClT)
    M=getMeasureCl(ClT,weight,noise)
    test(M,ClT)
    
def makeTestCl():
    if zeroPower:
        return claw.Cl(lmaxdl=(60,10))
    else:
        return claw.Cl(lmaxdl=(60,10),vals=[3.,2.,5.,10.,3.,1.])

def makeTestWindowNoise(ClT):
    Nside=ClT.Nside
    Npix=12*Nside**2
    theta,phi=hp.pix2ang(Nside,np.arange(Npix))
    noise=10.*(1.+theta/np.pi)
    if flatWeight:
        weight=1
    else:
        weight=1./noise**2
    return weight, noise

def getMeasureCl(ClT,weight,noise):
    if (isfile(pname)):
        print("Loading pickled version... (delete ",pname,"if changing params.")
        M=cp.load(open(pname))
    else:
        M=claw.MeasureCl(ClT,weight,noise,Ng=Ng)
        print ("Getting noise bias...")
        M.getNoiseBias()
        if flatWeight:
            M.setIdentityCoupling()
        else:
            print("Getting coupling matrix...")
            M.getCouplingMat()
        print("Getting covariance matrix...")
        M.getCovMat(ClT)
        cp.dump(M,open(pname,'w'),-1)

    return M

def test(M,ClT):    
    print("Testing Algorithm")
    res=[]
    chi2=[]
    chi2d=[]
    for cc in range(Ngtest):
        #generate test problem
        cls=ClT.Cl
        mp=hp.synfast(cls,ClT.Nside,verbose=False)
        mp+=np.random.normal(0.,M.Noise)
        ClM=M.getEstimate(mp)
        res.append(ClM.vals)
        chi2.append(ClM.chi2(ClT))
        chi2d.append(ClM.chi2diag(ClT))
        print("Go #",cc,"\r",end="")
    print ("\n")
    chi2=np.array(chi2)
    chi2d=np.array(chi2d)
    print(chi2.mean(), chi2.var())
    print(chi2d.mean(), chi2d.var())
    plt.hist(chi2,bins=50)

    plt.show()

    res=np.array(res)
    print ("Truth:",ClT.vals)
    print ("Mean:",res.mean(axis=0))
    print ("Err:",np.sqrt(np.cov(res,rowvar=False).diagonal()/Ngtest))
    print ("Oerr:",np.sqrt(ClM.cov.diagonal()/Ngtest))


if __name__=="__main__":
    main()
