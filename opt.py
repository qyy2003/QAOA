import numpy as np
import qutip as qt
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

import numpy as np
import random

sigma_z=np.array([[1,0],[0,-1]])
sigma_x=np.array([[0,1],[1,0]])

n=10
p=3

def generateGraph() :
    m = random.randint(1, n*(n-1)/2)
    edges = []
    for i in range(m) :
        x = random.randint(0, n-1)
        y = x
        while x == y:
            y = random.randint(0, n-1)
        edges.append([min(x, y), max(x, y)])
    return edges

graph = generateGraph()
np.save('grapy_in',graph)
graph = np.load('grapy_in.npy')
print(ar_load)
state0=np.array([1,0])
state1=np.array([0,1])

C=np.zeros((2**n,2**n))
s=np.zeros(2**n)
for edge in graph:
    tmp_C = 1
    for i in range(n):
        tmp_C = np.kron(tmp_C, sigma_z if i in edge else np.eye(2))
    C+=1/2*(np.eye(2**n)-tmp_C)
print(C.shape)

B=np.zeros((2**n,2**n))
for i in range(n):
    B+=np.kron(np.eye(2**i),np.kron(sigma_x,np.eye(2**(n-i-1))))

from scipy.linalg import expm,logm
def QAOA(gamma, beta):
    qs=np.ones(2**n)/np.sqrt(2**n)
    for i in range(p):
        qs=np.dot(expm(-1j*gamma[i]*C),qs)
        qs=np.dot(expm(-1j*beta[i]*B),qs)
    # print(qs)
    return np.matmul(qs.conj(),np.matmul(C,qs))

import numpy as np
from queue import PriorityQueue as PQ
import math
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy import optimize as opt

def objective(params):
    #print(params);
    gamma = params[0:p]
    gamma=np.clip(gamma, 0.01, 2*np.pi)
    beta = params[p:]
    # print("gamma:",gamma)
    beta=np.clip(beta, 0.01, np.pi)
    # print("beta",beta)
    return -QAOA(gamma, beta)

def printQAOA(gamma, beta):
    qs=np.ones(2**n)/np.sqrt(2**n)
    # print(np.dot(qs,qs))
    for i in range(p):
        qs=np.dot(expm(-1j*gamma[i]*C),qs)
        qs=np.dot(expm(-1j*beta[i]*B),qs)
    return np.matmul(qs.conj(),np.matmul(C,qs))

gamma = np.zeros(p)
beta = np.zeros(p)
grid = 7
max_ret = 0
def dfs(w = 0, i = 0):
    global max_ret
    if i == p:
        if w == 0:
            dfs(1, 0)
            return
        else :
            ret = QAOA(gamma, beta)
            max_ret = max(max_ret, ret)
            return
    if w == 0:
        for j in range(grid) :
            gamma[i] = np.pi * 2 / grid * j
            dfs(w, i+1)
    else :
        for j in range(grid) :
            beta[i] = np.pi * 2 / grid * j
            dfs(w, i+1)

## using https://github.com/hyperopt/hyperopt/wiki/FMin
import math
import numpy as np
from scipy.integrate import quad
from hyperopt import fmin, tpe, hp


def objective1(params):
    #print(params);
    gamma=[]
    beta=[]
    for i in range(p):
        gamma.append(params["g"+str(i)])
        beta.append(params["b"+str(i)])
    return -QAOA(gamma, beta)

space=[]
for i in range(p):
    space.append(("g"+str(i),hp.uniform('g'+str(i), 0, 2*np.pi)))
    space.append(("b"+str(i),hp.uniform('b'+str(i), 0, 2*np.pi)))
space=dict(space)
print(space)

print(graph)
gra = graphic(n, graph)
real_ans = gra.ask_min()
print(real_ans)

best = fmin(
    fn=objective1,
    space=space,
    algo=tpe.suggest,
    max_evals=500
)
print(best)


result=opt.basinhopping(objective,x0=np.array([np.random.rand(2*p)]), niter=100, niter_success=10, minimizer_kwargs={"method":"L-BFGS-B"}, disp=1)
print(result)

dfs(0, 0)
print(max_ret)
