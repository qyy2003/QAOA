import numpy as np
import warnings 
warnings.filterwarnings("ignore")
# import qutip as qt
import random
class graphic:
    def __init__(self,n,edges):
        self.n=n
        self.G=np.zeros((self.n,self.n))
        for[i,j] in edges:
            self.G[i][j] = self.G[i][j] + 1
    
    def ask_min(self):
        MASK=1<<self.n
        ans=0
        sum=0
        for mask in range(MASK):
            sum=0
            for i in range(self.n):
                for j in range(i+1,self.n):
                    if(((mask&(1<<i))>>i)+((mask&(1<<j))>>j)==1):
                        sum=sum+self.G[i][j]
            if(sum>ans):
                ans=sum
        return ans
    
    def ask_C(self):
        # opr=qt.zero(2)
        # print(opr)
        I=qt.tensor([qt.identity(2) for k in range(self.n)])
        for i in range(self.n):
            for j in range(i+1,self.n):
                print(qt.tensor([qt.identity(2) if (k != i and k != j) else qt.sigmaz() for k in range(self.n)]))
                opr=(I-qt.tensor([qt.identity(2) if (k != i and k != j) else qt.sigmaz() for k in range(self.n)]))*self.G[i][j]
        return opr

sigma_z=np.array([[1,0],[0,-1]])
sigma_x=np.array([[0,1],[1,0]])
global p
p=0
n=7
graph = np.load('grapy_in.npy')
state0=np.array([1,0])
state1=np.array([0,1])

C=np.zeros((2**n,2**n))
s=np.zeros(2**n)
for edge in graph:
    tmp_C = 1
    for i in range(n):
        tmp_C = np.kron(tmp_C, sigma_z if i in edge else np.eye(2))
    C+=1/2*(np.eye(2**n)-tmp_C)
# print(C.shape)

B=np.zeros((2**n,2**n))
for i in range(n):
    B+=np.kron(np.eye(2**i),np.kron(sigma_x,np.eye(2**(n-i-1))))

from scipy.linalg import expm,logm
def QAOA(gamma, beta):
    # print(p)
    qs=np.ones(2**n)/np.sqrt(2**n)
    for i in range(p):
        qs=np.dot(expm(-1j*gamma[i]*C),qs)
        qs=np.dot(expm(-1j*beta[i]*B),qs)
    # print(qs)
    return np.matmul(qs.conj(),np.matmul(C,qs))


from scipy import optimize as opt

def objective(params):
    #print(params);
    # print(p)
    gamma = params[0:p]
    gamma=np.clip(gamma, 0.01, 2*np.pi)
    beta = params[p:]
    # print("gamma:",gamma)
    beta=np.clip(beta, 0.01, np.pi)
    # print("beta",beta)
    return -QAOA(gamma, beta)

# p=int(input())
for pi in range(1,11):
    p=pi
    print("---\n\n P=",p)
    result=opt.basinhopping(objective,x0=np.array([np.random.rand(2*p)]), niter=100, niter_success=10, minimizer_kwargs={"method":"L-BFGS-B"}, disp=0)
    for i in range(10):
        print("-- Running on : "+str(i))
        result0=opt.basinhopping(objective,x0=np.array([np.random.rand(2*p)]), niter=100, niter_success=10, minimizer_kwargs={"method":"L-BFGS-B"}, disp=0)
        if(result['fun'].real>result0['fun'].real):
            result=result0
        np.save('result7_p/p='+str(pi),result['x'])
    print("\n",result)
    # np.save('p='+str(pi),result['x'])