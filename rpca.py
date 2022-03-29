import numpy as np
import denoiser 

# simple pcp-rpca implementation
def pcp(M, lam=None, mu=None, factor=1, tol=1e-3,maxit=1000,debug=True):
    # initialization
    m, n = M.shape
    unobserved = np.isnan(M)
    M[unobserved] = 0
    S = np.zeros((m,n))
    L = np.zeros((m,n))
    Lambda = np.zeros((m,n)) # the dual variable
 
    # parameter setting
    if mu is None:
        mu = 0.25/np.abs(M).mean()
    if lam is None:
        lam = 1/np.sqrt(max(m,n)) * float(factor)
        
    print('mu=',mu)
    print('lambda=',lam)
        
    # main
    for k in range(maxit):
        normLS = np.linalg.norm(np.concatenate((S,L), axis=1), 'fro')              
        # dS, dL record the change of S and L, only used for stopping criterion

        X = Lambda / mu + M
        # L - subproblem
        Y = X - S;
        dL = L;       
        U, sigmas, V = np.linalg.svd(Y, full_matrices=False);
        rank = (sigmas > 1/mu).sum()
        Sigma = np.diag(sigmas[0:rank] - 1/mu)
        L = np.dot(np.dot(U[:,0:rank], Sigma), V[0:rank,:])
        dL = L - dL
        
        # S - subproblem 
        Y = X - L
        dS = S
        S = denoiser.proxl1(Y, lam/mu) # softshinkage operator 
        dS = S - dS

        # Update Lambda (dual variable)
        Z = M - S - L
        Z[unobserved] = 0
        Lambda = Lambda + mu * Z;
        
        # stopping criterion
        RelChg = np.linalg.norm(np.concatenate((dS, dL), axis=1), 'fro') / (normLS + 1)
        if RelChg < tol: 
            break
            
        # debug
        if debug is True:
            print(k,':',RelChg)
    
    return L, S, k, rank



# stocRPCA utilities
def solve_proj(m,U,lambda1,lambda2,mu=None,tol=1e-3):
    # intialization
    n, p = U.shape
    v = np.zeros(p)
    s = np.zeros(n)
    I = np.identity(p)
    converged = False
    maxIter = 50 #np.inf
    k = 0
    # alternatively update
    UUt = np.linalg.inv(U.transpose().dot(U) + lambda1*I).dot(U.transpose())
    while not converged:
        k += 1
        vtemp = v
        # v = (U'*U + lambda1*I)\(U'*(m-s)) 
        v = UUt.dot(m - s) 
        stemp = s 
        
        # originally proxl1 
        s = denoiser.proxl1(m - U.dot(v), lambda2)
        
        #print('iter: ',k)
        stopc = max(np.linalg.norm(v - vtemp), np.linalg.norm(s - stemp))/n
        if stopc < tol or k > maxIter:
            converged = True
            #print(k)
            #print('inner ',k,'. stopc=',stopc)
    
    
    return v, s

def update_dicio(U, A, B, lambda1):
    m, q = U.shape
    A = A + lambda1*np.identity(q)
    for k in range(q):
        bk = B[:,k]
        uk = U[:,k]
        ak = A[:,k]
        temp = (bk - U.dot(ak))/A[k,k] + uk
        U[:,k] = temp/max(np.linalg.norm(temp), 1)
   
    return U
