import numpy as np
import matplotlib.pyplot as plt
import sys
from tm.base_models.base_model import BaseModel


class Gaussian(object):
    def __init__(self,n_gibbs=None,f_burn=0.1,min_k=0.25,max_k=0.25,names=None, min_abs_mean = 0):
        self.f_burn=f_burn
        self.n_gibbs=n_gibbs
        self.no_gibbs=False
        if self.n_gibbs is None:
            self.no_gibbs=True
            self.n_gibbs=0

        self.min_k=min_k
        self.max_k=max_k
        self.names=names
        self.min_abs_mean = min_abs_mean
        # real number of samples to simulate
        self.n_gibbs_sim=int(self.n_gibbs*(1+self.f_burn))
        self.p=1
        # to be calculated!
        self.gibbs_cov=None
        self.gibbs_mean=None
        self.mean=None
        self.cov=None
        self.cov_inv=None
        self.w=None
        self.w_norm=1

        assert self.max_k<=1 and self.max_k>=0,"max_k must be between 0 and 1"
        assert self.min_k<=1 and self.min_k>=0,"min_k must be between 0 and 1"
        assert self.max_k>=self.min_k,"max_k must be larger or equal than min_k"
        
    
    def view(self,plot_hist=True):
        if self.names is None:
            self.names=["x%s"%(i+1) for i in range(self.p)]
        if len(self.names)!=self.p:
            self.names=["x%s"%(i+1) for i in range(self.p)]                 
        print('** Gaussian **')
        print('Mean')
        print(self.mean)
        print('Covariance')
        print(self.cov)
        if self.gibbs_mean is not None:
            if plot_hist:
                for i in range(self.p):
                    plt.hist(self.gibbs_mean[:,i],density=True,alpha=0.5,label='Mean %s'%(self.names[i]))
                plt.legend()
                plt.grid(True)
                plt.show()
        if self.gibbs_cov is not None:
            if plot_hist:
                for i in range(self.p):
                    for j in range(i,self.p):
                        plt.hist(self.gibbs_cov[:,i,j],density=True,alpha=0.5,label='Cov(%s,%s)'%(self.names[i],self.names[j]))
                plt.legend()
                plt.grid(True)
                plt.show()          
            
    def estimate(self,y,**kwargs):      
        # Gibbs sampler
        assert y.ndim==2,"y must be a matrix"
        
        if self.no_gibbs:
            self.mean=np.mean(y,axis=0)
            self.mean[np.abs(self.mean) < self.min_abs_mean] = 0
            self.cov=np.cov(y.T)
            if self.cov.ndim==0:
                self.cov=np.array([[self.cov]])             
            # regularize
            self.cov=self.max_k*np.diag(np.diag(self.cov))+(1-self.max_k)*self.cov  
            if np.linalg.cond(self.cov) < 1/sys.float_info.epsilon:
                self.cov_inv=np.linalg.inv(self.cov)
                self.w=np.dot(self.cov_inv,self.mean)       
                self.w_norm=np.sum(np.abs(self.w))      
            else:
                print('Warning: singular cov...')
                self.cov_inv = np.diag(np.ones(self.cov.shape[0]))
                self.w = np.zeros(self.cov.shape[0])
                self.w_norm = 1
                #handle it
        else:
                
            n=y.shape[0]
            self.p=y.shape[1]       
            # compute data covariance
            c=np.cov(y.T)
            if c.ndim==0:
                c=np.array([[c]])       
            c_diag=np.diag(np.diag(c))
            # prior parameters
            m0=np.zeros(self.p) # mean prior center     
            V0=c_diag.copy() # mean prior covariance        
            S0aux=c_diag.copy() # covariance prior scale (to be multiplied later)
            # precalc
            y_mean=np.mean(y,axis=0)
            invV0=np.linalg.inv(V0)
            invV0m0=np.dot(invV0,m0)
            # initialize containers
            self.gibbs_cov=np.zeros((self.n_gibbs_sim,self.p,self.p))
            self.gibbs_mean=np.zeros((self.n_gibbs_sim,self.p))
            # initialize cov
            self.gibbs_cov[0]=c     
            # sample
            for i in range(1,self.n_gibbs_sim):
                # sample for mean
                invC=np.linalg.inv(self.gibbs_cov[i-1])
                Vn=np.linalg.inv(invV0+n*invC)
                mn=np.dot(Vn,invV0m0+n*np.dot(invC,y_mean))
                self.gibbs_mean[i]=np.random.multivariate_normal(mn,Vn)
                # sample from cov
                # get random k value (shrinkage value)
                k=np.random.uniform(self.min_k,self.max_k)
                n0=k*n/(1-k)
                S0=n0*S0aux
                v0=n0+self.p+1          
                vn=v0+n
                St=np.dot((y-self.gibbs_mean[i]).T,(y-self.gibbs_mean[i]))
                Sn=S0+St
                self.gibbs_cov[i]=invwishart.rvs(df=vn,scale=Sn) 
            self.gibbs_mean=self.gibbs_mean[-self.n_gibbs:]
            self.gibbs_cov=self.gibbs_cov[-self.n_gibbs:]
            self.mean=np.mean(self.gibbs_mean,axis=0)
            self.mean[np.abs(self.mean) < self.min_abs_mean] = 0
            self.cov=np.mean(self.gibbs_cov,axis=0)
            self.cov_inv=np.linalg.inv(self.cov)
            self.w=np.dot(self.cov_inv,self.mean)
            self.w_norm=np.sum(np.abs(self.w))

    def get_weight(self,normalize=True,**kwargs):
        if normalize:
            return self.w/self.w_norm
        else:
            return self.w




class ConditionalGaussian(object):
    def __init__(self,n_gibbs=None,f_burn=0.1,min_k=0.25,max_k=0.25,kelly_std=2,max_w=1,bias_reduction=0):
        self.n_gibbs=n_gibbs
        self.no_gibbs=False
        if self.n_gibbs is None:
            self.no_gibbs=True
        self.f_burn=f_burn
        self.max_k=max_k
        self.bias_reduction = bias_reduction
        self.kelly_std=kelly_std
        self.max_w=max_w
        self.min_k=min_k
        self.g=None
        # to calculate after Gaussian estimate
        self.my=None
        self.mx=None
        self.Cyy=None
        self.Cxx=None
        self.Cyx=None
        self.invCxx=None
        self.pred_gain=None
        self.cov_reduct=None
        self.pred_cov=None 
        self.prev_cov_inv=None
        self.w_norm=1

    
    def view(self,plot_hist=True):
        if self.g is not None:
            self.g.view(plot_hist=plot_hist)
    
    def estimate(self,y,x,**kwargs): 
        x=x.copy()
        y=y.copy()      
        if x.ndim==1:
            x=x[:,None]
        if y.ndim==1:
            y=y[:,None]     
        p=y.shape[1]
        q=x.shape[1]        
        z=np.hstack((y,x))
        names=[]
        for i in range(p):
            names.append("y%s"%(i+1))
        for i in range(q):
            names.append("x%s"%(i+1))
        self.g=Gaussian(self.n_gibbs,self.f_burn,self.min_k,self.max_k,names=names)
        self.g.estimate(z)
        # extract distribution of y|x from the estimated covariance
        y_idx=np.arange(p)
        x_idx=np.arange(p,p+q)      
        self.my=self.g.mean[y_idx]
        self.mx=self.g.mean[x_idx]
        self.Cyy=self.g.cov[y_idx][:,y_idx]
        self.Cxx=self.g.cov[x_idx][:,x_idx]
        self.Cyx=self.g.cov[y_idx][:,x_idx]
        self.invCxx=np.linalg.inv(self.Cxx)
        self.pred_gain=np.dot(self.Cyx,self.invCxx)
        self.cov_reduct=np.dot(self.pred_gain,self.Cyx.T)
        self.pred_cov=self.Cyy-self.cov_reduct
        self.pred_cov_inv=np.linalg.inv(self.pred_cov)
        # compute normalization
        x_move=np.sqrt(np.diag(self.Cxx))*self.kelly_std
        self.w_norm=np.sum(np.abs(np.dot( self.pred_cov_inv , self.my + np.dot(np.abs(self.pred_gain),x_move+self.mx) )))

    def predict(self,xq):
        return self.my+np.dot(self.pred_gain,xq-self.mx)
    
    def expected_value(self,xq):
        return predict(xq)
    
    def covariance(self,xq):
        return self.pred_cov
    
    def get_weight(self,xq,normalize=True,**kwargs):
        if normalize:
            if not hasattr(self, 'bias_reduction'):
                self.bias_reduction = 0
            w = np.dot(self.pred_cov_inv,self.predict(xq)) - self.bias_reduction*np.dot(self.pred_cov_inv,self.my)          
            w /= self.w_norm
            d=np.sum(np.abs(w))
            if d>self.max_w:
                w/=d
                w*=self.max_w
            return w            
        else:
            return np.dot(self.pred_cov_inv,self.predict(xq))
