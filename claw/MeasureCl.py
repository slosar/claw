#
# Object that actually measures Cls
#

from __future__ import print_function, division

import numpy as np
import healpy as hp
import scipy.linalg as la
import copy
class MeasureCl:
    def __init__(self, Cl, Weight, Noise,Ng=1000,Noise2=None,Weight2=None, ignorem0=False,narowcoupling=False):
        """
          When initializing, set it up with Cl object for binning
          and cov matrix prediction.
          Weight is the weight map. It can be arbirary, but Weight=1/Noise**2 is optimal.
          Ng is the number of random samples it takes to get coupling matrix. It needs to be
          surprisingly large.
        """
          
        self.Cl=copy.deepcopy(Cl)
        self.ignorem0=ignorem0
        self.narowcoupling=narowcoupling
        self.Weight=Weight
        self.Noise=Noise
        self.Nside=Cl.Nside
        self.lmax=Cl.lmaxs#self.Nside*3-1
        self.Npix=12*self.Nside**2
        self.nbins=self.Cl.nbins
        self.mbins=self.nbins+1 ## mbins includes the edge bin that we are throwing away
        self.m0indices=np.arange(self.lmax+1)    #first 3*Nside elements corespond to m=0 modes (for ignoring m=0 modes)
        self._getbinlist()
        self.Ng=Ng
        
        if Noise2 is not None:
            self.Noise2=Noise2
            
        if Weight2 is not None:
            self.Weight2=Weight2

    def _getbinlist(self):
        lar=[]
        for m in range(0,self.lmax+1):
            for l in range (m,self.lmax+1):
                lar.append(l)
        ## now get mapping from lmax to bins
        self.binlist=self.Cl.ndx[lar]
        self.bnorm=1./(np.bincount(self.binlist)-self.Cl.dl)
        if self.ignorem0:           # adjusting the binlist to fit together with almsq from _getIM when ignoring m=0 modes
            self.binlist=np.delete(self.binlist,self.m0indices)

    def _getIM(self,mp, addN=False):
        if addN:
            mp+=np.random.normal(0,self.Noise)
        mp*=self.Weight
        almsq=abs(hp.map2alm(mp,lmax=self.lmax)**2)
        if self.ignorem0:
            almsq=np.delete(almsq,self.m0indices)
        return self.bnorm*np.bincount(self.binlist,weights=almsq)
    
    def _getcrossIM(self,mp1,mp2, addN=False):
        if addN:
            mp1+=np.random.normal(0,self.Noise)
            mp2+=np.random.normal(0,self.Noise2)
        mp1*=self.Weight
        mp2*=self.Weight2
        cross_almsq=(hp.map2alm(mp1,lmax=self.lmax)*np.conjugate(hp.map2alm(mp2,lmax=self.lmax))).real
        if self.ignorem0:
            cross_almsq=np.delete(cross_almsq,self.m0indices)
        return self.bnorm*np.bincount(self.binlist,weights=cross_almsq)

    def getNoiseBias(self):
        nv=[]
        for cc in range(self.Ng):
            nv.append(self._getIM(np.zeros(self.Npix),addN=True))
        nv=np.array(nv)
        self.nbias=nv.mean(axis=0)
        self.ncov=np.cov(nv,rowvar=False)

    def setIdentityCoupling(self):
        self.coupmat=np.identity(self.mbins-1)
        self.icoupmat=np.identity(self.mbins-1)
        
    def getCouplingMat(self):
        cmat=np.zeros((self.mbins-1,self.mbins-1))
        irange=self.mbins-1
        for i in range(irange):
            for cc in range(self.Ng):
                clx=np.zeros(self.lmax+1)
                if (i==self.nbins):
                    clx[self.Cl.lmax[-1]:]=1.0
                else:
                    clx[self.Cl.lmin[i]:self.Cl.lmax[i]]=1.0
                m=hp.synfast(clx,self.Nside,verbose=False)
                cmat[i,:]+=self._getIM(m)
        self.coupmat=cmat/self.Ng
        self.icoupmat=la.inv(self.coupmat)
        
    def getcrossCouplingMat(self):
        cmat=np.zeros((self.mbins-1,self.mbins-1))
        irange=self.mbins-1
        for i in range(irange):
            for cc in range(self.Ng):
                clx=np.zeros(self.lmax+1)
                if (i==self.nbins):
                    clx[self.Cl.lmax[-1]:]=1.0
                else:
                    clx[self.Cl.lmin[i]:self.Cl.lmax[i]]=1.0
                m=hp.synfast(clx,self.Nside,verbose=False)
                cmat[i,:]+=self._getcrossIM(m,m)
        self.coupmat=cmat/self.Ng
        if self.narowcoupling:
            for j in range(irange):
                indx=np.arange(self.nbins)
                indx=np.delete(indx,np.array([j-2,j-1,j,j+1,j+2]))
                self.coupmat[j,:][indx]=0
        self.icoupmat=la.inv(self.coupmat)

    def getCovMat(self,thCl=None):
        if not hasattr(self,"coupmat"):
            self.getCouplingMat()
        if (hasattr(self,"nbias") and (thCl is None)):
            tcov=self.ncov
        else:
            clx=thCl.Cl
            nv=[]
            for cc in range(self.Ng):
                m=hp.synfast(clx,self.Nside,verbose=False)
                r=self._getIM(m,addN=True)
                nv.append(r)

            nv=np.array(nv)
            tcov=np.cov(nv,rowvar=False)
        cov=np.dot(self.icoupmat,np.dot(tcov,self.icoupmat.T))
        self.Cl.setCov(cov[:self.nbins, :self.nbins])
        
    def getcrossCovMat(self,thCl=None):
        if not hasattr(self,"coupmat"):
            self.getcrossCouplingMat()
        
        clx=thCl.Cl
        nv=[]
        for cc in range(self.Ng):
            m=hp.synfast(clx,self.Nside,verbose=False)
            r=self._getcrossIM(m,m,addN=True)
            nv.append(r)

        nv=np.array(nv)
        tcov=np.cov(nv,rowvar=False)
        cov=np.dot(self.icoupmat,np.dot(tcov,self.icoupmat.T))
        self.Cl.setCov(cov[:self.nbins, :self.nbins])

    def getEstimate(self,mp):
        if not hasattr(self,"coupmat"):
            self.getCouplingMat()
        if not hasattr(self,"nbias"):
            self.getNoiseBias()
        v=self._getIM(mp)
        self.Cl.setVals(np.dot(self.icoupmat,v-self.nbias))
        return self.Cl
    
    
    def getcrossEstimate(self,mp1,mp2):
        if not hasattr(self,"coupmat"):
            self.getcrossCouplingMat()
        v=self._getcrossIM(mp1,mp2)
        self.Cl.setVals(np.dot(self.icoupmat,v))
        return self.Cl
