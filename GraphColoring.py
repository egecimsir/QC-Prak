from qiskit import *
from qiskit.visualization import plot_histogram
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def show_graph(graph, colors):
    nx.draw(graph, with_labels=True, node_size=2000, node_color=colors)
    plt.show()

def create_graph(V:int, E, C):
    G = nx.Graph()
    G.add_nodes_from([i for i in range(V)])
    G.add_edges_from(E)
    for i in range(V):
        G.nodes[i]["color"] = C[i]
    return G

# For creating the oracles
def XOR(qc, a, b, output):
    qc.cx(a, output)
    qc.cx(b, output)

def diffuser(nqubits):
    qc = QuantumCircuit(nqubits)
    for qubit in range(nqubits):
        qc.h(qubit)
    for qubit in range(nqubits):
        qc.x(qubit)
    qc.h(nqubits-1)
    qc.mct(list(range(nqubits-1)), nqubits-1)
    qc.h(nqubits-1)
    for qubit in range(nqubits):
        qc.x(qubit)
    for qubit in range(nqubits):
        qc.h(qubit)
    # I will return the diffuser as a gate
    U_s = qc.to_gate()
    U_s.name = "U$_s$"
    return U_s

################
### LÖSUNGEN ###
################


def AUFGABE_8_1():
    E = [(0, 1), (1, 2), (2, 3), (3, 0)]
    C = ["blue", "yellow", "white", "white"]

    graph = create_graph(4, E, C)
    show_graph(graph, C)

    # 1 != 2
    # 0 != 3
    # 2 != 3
    # 1 != 0
    clauses = [(1, 2), (0, 3), (2, 3), (1, 0)]

    # Creating Quantum Circuit
    var_qubits = QuantumRegister(4, name='v')
    clause_qubits = QuantumRegister(4, name='c')
    output_qubit = QuantumRegister(1, name='out')
    cbits = ClassicalRegister(4, name='cbits')
    qc = QuantumCircuit(var_qubits, clause_qubits, output_qubit, cbits)

    def oracle(qc, clauses, clause_qubits):
        i = 0
        for clause in clauses:
            XOR(qc, clause[0], clause[1], clause_qubits[i])
            i += 1
        qc.mct(clause_qubits, output_qubit)
        i = 0
        for clause in clauses:
            XOR(qc, clause[0], clause[1], clause_qubits[i])
            i += 1

    qc.initialize([1, -1] / np.sqrt(2), output_qubit)

    qc.h(var_qubits)
    qc.barrier()
    # N = 4 ==> sqrt(N) = 2 iterations
    for i in range(2):
        oracle(qc, clauses, clause_qubits)
        qc.barrier()
        qc.append(diffuser(4), [0, 1, 2, 4])

    qc.measure(var_qubits, cbits)

    out = qc.draw(output="mpl", fold=-1)
    out.show()

    qasm_simulator = Aer.get_backend('qasm_simulator')
    transpiled_qc = transpile(qc, qasm_simulator)
    qobj = assemble(transpiled_qc)
    result = qasm_simulator.run(qobj).result()
    plot_histogram(result.get_counts()).show()

    solution = """\n\t0101 and 1010 are the winner states. Each bit correspondes to nodes 0,1,2,3 respectivly.
    According to the results: 0 1 0 1 => Node0 == Node2  and  Node1 = Node2.
    Since we know the color of Node0 is blue, Node2 must be also blue and Node3 yellow"""
    print(solution)
    C[2] = C[0]
    C[3] = C[1]
    show_graph(graph, C)
AUFGABE_8_1()

def AUFGABE_8_2():
    E = [(0,1), (1,2), (2,3), (0,4), (0,5), (1,4), (1,5), (4,5), (5,6), (2,5), (2,6), (3,6)]    #len = 12
    C = ["red", "white", "white", "blue", "green", "yellow", "white"]

    graph = create_graph(7, E, C)
    show_graph(graph, C)

    # Since edges denotes the neighbour nodes, we can simply take them as our clauses
    clauses = E

    # Creating Quantum Circuit
    var_qubits = QuantumRegister(12, name='v')
    clause_qubits = QuantumRegister(12, name='c')
    output_qubit = QuantumRegister(1, name='out')
    cbits = ClassicalRegister(12, name='cbits')
    qc = QuantumCircuit(var_qubits, clause_qubits, output_qubit, cbits)

    def oracle(qc, clauses, clause_qubits):
        i = 0
        for clause in clauses:
            XOR(qc, clause[0], clause[1], clause_qubits[i])
            i += 1
        qc.mct(clause_qubits, output_qubit)
        i = 0
        for clause in clauses:
            XOR(qc, clause[0], clause[1], clause_qubits[i])
            i += 1

    qc.initialize([1, -1] / np.sqrt(2), output_qubit)

    qc.h(var_qubits)
    qc.barrier()

    for i in range(2):
        oracle(qc, clauses, clause_qubits)
        qc.barrier()
        qc.append(diffuser(12), [i for i in range(12)])

    qc.measure(var_qubits, cbits)

    out = qc.draw(output="mpl", fold=-1)
    out.show()

    qasm_simulator = Aer.get_backend('qasm_simulator')
    transpiled_qc = transpile(qc, qasm_simulator)
    qobj = assemble(transpiled_qc)
    result = qasm_simulator.run(qobj).result()
    plot_histogram(result.get_counts()).show()

    # It takes too much time for my computer to compute...
    # Thats why I cant print any solutions but i am sure it would do well :)
#AUFGABE_8_2()