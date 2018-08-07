#
# Object containing Cell measurements
#

from __future__ import print_function, division
import numpy as np
import scipy.linalg as la

class Cl:
    def __init__(self, lmaxdl=(100,10), lbins=None, vals=None, cov=None, Nside=None):
        """
        You need to decide on binning when creating this object.
        Specify either via lmax and dl or explicitly set lbins and the bin
        edges will be adjusted automatically.
        """
        if lbins is not None:
            lcent=np.array(lbins)
            lmin=[1]+list((0.5*(lcent[1:]+lcent[:-1])).astype(int))
            lmax=lmin[1:]+[lcent[-1]+lcent[-1]-lmin[-1]]
        else:
            lmax,dl=lmaxdl          #actualy lmax+1 has to be entered (number of l values in the original power spectrum)
            nb=int(lmax/dl)
            lmin=dl*np.arange(nb)
            lmax=dl*(1+np.arange(nb))
            lcent=(lmin+lmax-1)/2.
            #if (dl>1):
            #    lmin[0]=1
        # now adjust last bin until we have full
        # ell coverage to avoid aliasing
        self.dl=lmaxdl[1]
        self.lmaxs=lmaxdl[0]-1
        self.lmin=lmin
        self.lmax=lmax
        if Nside is not None:
            self.Nside=Nside
        else:
            self.Nside=2
            while (3*(self.Nside+1)<lmax[-1]):
                self.Nside*=2
        self.ells=lcent
        self.nbins=len(lcent)
        self.vals=np.zeros(self.nbins)
        if vals is not None:
            self.vals=vals
        if cov is not None:
            self.setCov(cov)

        self.ndx=np.zeros((self.lmaxs+1),np.int)+self.nbins ## non-bins go to the lastone
        self.Cl=np.zeros(self.lmaxs+1)
        for i,v,l1,l2 in zip (np.arange(self.nbins),self.vals,self.lmin, self.lmax):
            self.ndx[l1:l2]=i
            self.Cl[l1:l2]=v

    def setCov(self,cov):
        assert(cov.shape==(self.nbins,self.nbins))
        self.cov=cov
        self.icov=la.inv(cov)

    def setVals(self,vals):
        self.vals=vals
        for v,l1,l2 in zip (self.vals,self.lmin, self.lmax):
            self.Cl[l1:l2]=v
            
    def chi2(self, theo):
        diff=theo.vals-self.vals
        return np.dot(diff,np.dot(self.icov,diff))

    def chi2diag(self,theo):
        return ((theo.vals-self.vals)**2/self.cov.diagonal()).sum()

