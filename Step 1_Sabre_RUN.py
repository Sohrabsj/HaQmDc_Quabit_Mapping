#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 13:02:21 2025

@author: sohrab
"""

# IBM runtime import
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

#from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from functions import qasm_to_qiskit, circuit_to_qiskit_code, reset_aware_depth, fidelity_from_two_counts
from qasm import qasm_code
from qiskit.visualization import plot_circuit_layout

# --------------------------------------------------------------------
# === EMBEDDED CREDENTIALS (as requested) ===

BACKEND_NAME = "ibm_brisbane"
IBM_INSTANCE = 'crn:v1:bluemix:public:quantum-computing:us-east:a/715ce035d2a04044a4fe6d9f0ea70139:46a3f9a4-e9d1-463b-b44e-1bf99687ba53::'
IBM_TOKEN = 'fjwoqKLbXnVbzbEH-6ZOOdQgt_pvC8J6gMn8FQwuGMhD'
QiskitRuntimeService.save_account(token=IBM_TOKEN, instance=IBM_INSTANCE, overwrite=True)
service = QiskitRuntimeService()
backend = service.backend(BACKEND_NAME)
print(backend)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
qasmfile = "QAOA_6q"
shots = 2048
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

file = open("Results/"+qasmfile+"_Sabre.txt", "w")


# ---  Original qiskit circuit from qasm, If you have  a qiskit code for a circuit, you can skip this block ---and use your circuit directly
#qc = qasm_to_qiskit(qasm_code)  

qc_qasm = QuantumCircuit.from_qasm_file("QASMfiles/"+qasmfile+".qasm")
# Decompose only CCX gates, leave others intact
qc = qc_qasm.decompose(gates_to_decompose=['ccx'])


n = qc.num_qubits  # number of qubits in the circuit
# Ensure the circuit has enough classical bits
if len(qc.clbits) < n:
    qc.add_register(ClassicalRegister(n - len(qc.clbits)))
# Measure all qubits into all classical bits

#num_qubits_to_measure = min(len(qc.qubits), len(qc.clbits))
#qc.measure(qc.qubits[:num_qubits_to_measure], qc.clbits[:num_qubits_to_measure])
qc.measure(qc.qubits, qc.clbits)
code_qiskit = circuit_to_qiskit_code(qc)

#print("Original Circuit Qiskit Code:")
#print(code_qiskit)

from IPython.display import display
display(qc.draw('mpl', fold=-1))

simulator = AerSimulator()
job = simulator.run(qc, shots=shots)
result = job.result()
counts1 = result.get_counts(qc)
#print(f"Measurement counts Simulator: {counts}")
for outcome, count in counts1.items():
    #print(f"{outcome}: {count}")
    file.write(f"SIM: {outcome}: {count} \n")

# Submit the job using the SamplerV2 primitive
sampler = SamplerV2(backend)  # or backend_real

from qiskit.transpiler import CouplingMap
coupling_edges = backend.configuration().coupling_map
coupling_map = CouplingMap(coupling_edges)


qubit_count = qc.num_qubits 


transpiled_qc = transpile(
    qc,
    backend,
    coupling_map = coupling_map,
    #scheduling_method="asap",
    #optimization_level= 2,
    routing_method='sabre'      # Use SABRE for qubit mapping
)



# Get the initial layout
layout = transpiled_qc._layout.initial_layout
# Extract physical qubits of the main 'q' register
q_list = [
    phys for phys, qubit in layout.get_physical_bits().items()
    if f"QuantumRegister({qubit_count}, 'q')" in str(qubit)
]
init_map = {i: q for i, q in enumerate(q_list)}  #{0: 0, 1: 1, 2: 2, 3: 3}

# Print the list
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
'''duration_dt = transpiled_qc.duration
print("Duration (in dt units):", duration_dt)
dt = backend.configuration().dt  # seconds per dt
duration_sec = duration_dt * dt
print("Duration (in seconds):", duration_sec)'''

print("Standard depth:", transpiled_qc.depth())
print("Reset-aware depth:", reset_aware_depth(transpiled_qc))


print("Logical -> Physical qubit mapping:", init_map)
num_cx = sum(1 for inst in qc.data if inst.operation.name == "cx")
print("Number of CX gates:", num_cx)

num_ecr = sum(1 for inst in transpiled_qc.data if inst.operation.name == "ecr")
print("Number of ECR gates:", num_ecr)

print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print("Operations by type:", transpiled_qc.count_ops())
print("Total ops:", transpiled_qc.size())
print("Circuit depth:", transpiled_qc.depth())
print("Circuit 2Q gates depth:", transpiled_qc.depth(lambda x: x.operation.num_qubits == 2))
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")


from qiskit.visualization import plot_circuit_layout, plot_gate_map
from IPython.display import display
display(plot_circuit_layout(transpiled_qc, backend))
#display(plot_gate_map(backend))

job = sampler.run([transpiled_qc], shots=shots)
result = job.result()[0]

# Combine data from all classical registers
combined_data = result.join_data()

# Get the counts of all measured bitstrings
counts2 = combined_data.get_counts()

# Display the measurement counts
print("Measurement counts:")

for outcome, count in counts2.items():
    #print(f"{outcome}: {count}")
    file.write(f"IBM: {outcome}: {count} \n")

file.close()

fidelity, mse = fidelity_from_two_counts(counts1, counts2)
print("Fidelity =", fidelity)
print("MSE Loss =", mse)