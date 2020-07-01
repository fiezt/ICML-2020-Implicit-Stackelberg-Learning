import torch
from torch import autograd
import numpy as np
import scipy.sparse.linalg
import time
np.set_printoptions(precision=2)
    
    
class JacobianVectorProduct(scipy.sparse.linalg.LinearOperator):
    def __init__(self, grad, params, regularization=0):
        if isinstance(grad, (list, tuple)):
            grad = list(grad)
            for i, g in enumerate(grad):
                grad[i] = g.view(-1)
            self.grad = torch.cat(grad)
        elif isinstance(grad, torch.Tensor):
            self.grad = grad.view(-1)

        nparams = sum(p.numel() for p in params)
        self.shape = (nparams, self.grad.size(0))
        self.dtype = np.dtype('Float32')
        self.params = params
        self.regularization = regularization

    def _matvec(self, v):
        v = torch.Tensor(v)
        if self.grad.is_cuda:
            v = v.cuda()
        hv = autograd.grad(self.grad, self.params, v, retain_graph=True, allow_unused=True)
        _hv = []
        for g, p in zip(hv, self.params):
            if g is None:
                g = torch.zeros_like(p)
            _hv.append(g.contiguous().view(-1))
        if self.regularization != 0:
            hv = torch.cat(_hv) + self.regularization*v
        else:
            hv = torch.cat(_hv) 
        return hv.cpu()
    
    def _matmat(self, X):

        return np.hstack([self._matvec(col) for col in X.T])

    
class SchurComplement(scipy.sparse.linalg.LinearOperator):
    """
    e.g.
    
    M = SchurComplement(A,B,C,D)
    v = torch.tensor(size=D.shape[0])
    q = M(v)
    eigs = eigvals(M)
  
    `N = np.randn((d,d))
    eigs = eigvals(N)
    
    
    """
    
    def __init__(self, A, B, C, D, tol_gmres=1e-6, precise=False):
        self.operator = [[A,B], [C,D]]
        self.shape = A.shape
        self.config = {'tol_gmres': tol_gmres}
        self.dtype = np.dtype('Float32')
        self.precise = precise
        
    def _matvec(self, v): 
        
        (A,B),(C,D) = self.operator

        u = C(v)
        
        if self.precise:
            w, status = scipy.sparse.linalg.gmres(D, u, tol=self.config['tol_gmres'], restart=D.shape[0])
            assert status == 0
        else:
            w, status = scipy.sparse.linalg.cg(D, u, maxiter=20)
        
        self.w = w

        p = A(v) - B(w)
        
        return p
    
    def _matmat(self, X):

        return np.hstack([self._matvec(col) for col in X.T])
    
class SchurComplement2(scipy.sparse.linalg.LinearOperator):
    def __init__(self, A, B, C, D, Dreg=None, tol_gmres=1e-6, precise=False):
        self.operator = [[A,B], [C,D]]
        self.Dreg = Dreg 
        self.shape = A.shape
        self.config = {'tol_gmres': tol_gmres}
        self.dtype = np.dtype('Float32')
        self.precise = precise
        
    def _matvec(self, v): 
        """
        A = D11f1 
        B = D12f1
        C = D21f2
        D = D22f2
        Dreg = D22f2 + etaI
                
        Dr = -(D22f2)^-1 * D21f2 = schur(0 I; D21f2 D22f2+etaI)
        
        [operator]v = [D11f1 + 2*D12f1*Dr + Dr^T D22f1 Dr ]v
                 = [D11f1 - 2*D12f1*(D22f2 + etaI)^-1*D21f2 + (D21f2^T*(D22f2+ etaI)^-1*D22f1*(D22f2 + etaI)^-1*D21f2] v
                 = [A - 2 * B * Dreg^-1 * C    + C^T * Dreg^-1 * E * Dreg^-1 * C]v
        (zerosum)= [A - 2 * B * Dreg^-1 * C    + (-1)(-1) B * Dreg^-1 * D * Dreg^-1 * C] v
                 = A - 2 * B * Dreg^-1 * u     + B * Dreg^-1 * D * Dreg^-1 * u
                 = A - 2 * B(w)                + B * Dreg^-1 * D(w)
                 = A - 2 * B(w)                + B * Dreg^-1 * s
                 = A - 2 * B(w)                + B(t)
        """
        
        (A,B),(C,D) = self.operator
        Dreg = self.Dreg

        u = C(v)
        
        if self.precise:
            w, status = scipy.sparse.linalg.gmres(Dreg, u, tol=self.config['tol_gmres'], restart=Dreg.shape[0])
            assert status == 0
        else:
            w, status = scipy.sparse.linalg.cg(Dreg, u, maxiter=20)
                
        s = D(w)
        
        if self.precise:
            t, status = scipy.sparse.linalg.gmres(Dreg, s, tol=self.config['tol_gmres'], restart=Dreg.shape[0])
            assert status == 0
        else:
            t, status = scipy.sparse.linalg.cg(Dreg, s, maxiter=20)
        
        p = A(v) - 2*B(w) + B(t)
        
        return p
    
    
    def _matmat(self, X):

        return np.hstack([self._matvec(col) for col in X.T])
    
    
    
