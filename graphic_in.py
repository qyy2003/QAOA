
import random
import numpy as np
n=7
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
        for mask in range(MASK):
            sum=0
            for i in range(self.n):
                for j in range(i+1,self.n):
                    if(((mask&(1<<i))>>i)+((mask&(1<<j))>>j)==1):
                        sum=sum+self.G[i][j]
            if(sum==ans):
                print(' %#x'%mask)
        return ans
    
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
gra=graphic(n,graph )
print(gra.ask_min())
np.save('grapy_in',graph)