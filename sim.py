# set & connect IBM Quantum account 
from qiskit import IBMQ
My_token = 'xxxx'#(IBM Quantum account token)
IBMQ.save_account(My_token)
provider = IBMQ.load_account()

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, Aer, execute, transpile
from qiskit.visualization import plot_histogram, array_to_latex
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)
# beta implement
def create_HD(parameter,G_info):
    nqubits = len(G_info.nodes())
    beta = parameter
    qc_mix = QuantumCircuit(nqubits)
    for i in range(0, nqubits):
        qc_mix.rx(beta*2, i)
    return qc_mix

# gamma implement
def create_Hp(parameter,G_info):
    nqubits = len(G_info.nodes())
    gamma = parameter
    qc_mix = QuantumCircuit(nqubits)
    edge = G_info.edges()
    for pair_nodes in edge:
        # print( pair_nodes)
        i = pair_nodes[0]
        j = pair_nodes[1]
        qc_mix.cx(i,j)
        qc_mix.rz(-1*gamma*G_info.edges[i,j]['weight'],j)
        qc_mix.cx(i,j)
        qc_mix.barrier()
    return qc_mix
            
# create QAOA quantum circuit
def create_qaoa(parameters_info,G):
    p = int(len(parameters_info)/2)
    nqubits = len(G.nodes())
    circ = QuantumCircuit(nqubits)
    for i in range(0,nqubits):
        circ.reset(i)
        circ.h(i)
    G_info = G
   
    parameter_Hp = parameters_info[0:p]
    parameter_HD = parameters_info[p:]
   
    for i in range(0,p):
        parameter1 = parameter_HD[i]
        circ1 = create_HD(parameter1,G_info)
        parameter2 = parameter_Hp[i]
        circ2 = create_Hp(parameter2,G_info)
        new_circ = circ2.compose(circ1)
        circ = circ.compose(new_circ) 
    qaoa_circ = circ
    qaoa_circ.measure_all()
    return qaoa_circ

## to init Graph
import matplotlib.pyplot as plt

global G
G = nx.Graph()
n=7
for i in range(n):
    G.add_node(i)
graph = np.load('grapy_in.npy')
for edge in graph:
    if(G.has_edge(edge[0],edge[1])):
        G.edges[edge[0],edge[1]]['weight']=G.edges[edge[0],edge[1]]['weight']+1
    else:
        G.add_edge(edge[0],edge[1], weight=1)
        
## to simulate the quantum 
An_WoN=[]
An_WN=[]
# get real quantum machine error params
backend = provider.get_backend('ibm_nairobi')
noise_model = NoiseModel.from_backend(backend)
coupling_map = backend.configuration().coupling_map
basis_gates = noise_model.basis_gates

for p in range(1,7):
    parameters=np.load("result7_p/p={}.npy".format(str(p)))
    print(parameters)
    circ = create_qaoa(parameters,G)
    ##simulate without Error
    sim = Aer.get_backend('qasm_simulator')         #for 1024 shots
    # result = sim.run(circ,shots=100000).result()  #for 100000 shots
    result = sim.run(circ).result()
    Results = result.get_counts()
    ex=result.results[0]
    print(p,str((ex.data.counts['0x1a']+ex.data.counts['0x65'])/ex.shots*100)+"%")
    An_WoN.append((ex.data.counts['0x1a']+ex.data.counts['0x65'])/ex.shots*100)

    # simulate with real quantum machine error params
    circ = create_qaoa(parameters,G)
    result = execute(circ, Aer.get_backend('qasm_simulator'),
                     coupling_map=coupling_map,
                     basis_gates=basis_gates,
                     noise_model=noise_model).result()                #for 1024 shots
                     # noise_model=noise_model,shots=100000).result() #for 100000 shots
    counts = result.get_counts(0)
    ex=result.results[0]
    print(p,str((ex.data.counts['0x1a']+ex.data.counts['0x65'])/ex.shots*100)+"%")
    An_WN.append((ex.data.counts['0x1a']+ex.data.counts['0x65'])/ex.shots*100)
    # Plot p output
    plot_histogram([counts,Results],legend = ['With Noise', 'Without Noise'],sort='desc',number_to_keep=5,figsize=(12,5)).savefig("figure/[shots=1024]P={}".format(p),bbox_inches='tight')
    
## plot the Accuracy-p fig
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1, 7, 1)
print(x)
y1 = np.exp(-x)
y2 = np.log(x)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, An_WN,'r',marker='v',label="With Noise");
ax1.legend(loc=1)
ax1.set_ylabel('Accuracy With Noise(%)');
ax1.set_xlabel('p');
ax2 = ax1.twinx() 
ax2.plot(x, An_WoN, 'g',marker='x',label = "Without Noise")
ax2.legend(loc=2)
ax2.set_xlim([0.5,6.5]);
ax2.set_ylabel('Accuracy Without Noise(%)');
ax2.set_xlabel('p');
fig.savefig("figure/[shots=1024]Accuracy-P")
plt.show()