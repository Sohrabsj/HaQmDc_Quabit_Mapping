import networkx as nx
# IBM runtime import
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from functions import fidelity_from_two_counts, qasm_to_qiskit, circuit_to_qiskit_code, compact_qubits, replay_to_qiskit, reset_aware_depth
from dynamic_circuit import dynamic_circuit, reorder_instructions
from qasm import qasm_code
from Beam_search import qiskit_instrs_to_immutable_nodes, build_dag_from_nodes, initialize_front_layer, get_distance_matrix
from Beam_search import get_calibration_from_ibm, BeamSabre, count_stats
from qiskit.transpiler import CouplingMap
from IPython.display import display
from qiskit.visualization import plot_circuit_layout

# === EMBEDDED CREDENTIALS (as requested) ===
BACKEND_NAME = "ibm_brisbane"
IBM_INSTANCE = 'crn:v1:bluemix:public:quantum-computing:us-east:a/305b501baafd410c9bc1fea7ae4f63d1:11742248-c8e1-41e4-aee6-006dcfddac66::'
IBM_TOKEN = 'tsbihBSOgHrKuPA0CAUNUPxcMUFe-xNxLwVXZDrlp1oA'
service = QiskitRuntimeService()
backend = service.backend(BACKEND_NAME)
print(backend)    

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
qasmfile = "test"
shots = 2048
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

file = open("Results/"+qasmfile+"_Dyn.txt", "w")

# fetch calibration (optional)
calibration = None
props = backend.properties()
if IBM_INSTANCE and IBM_TOKEN and BACKEND_NAME:
    try:
        print("Fetching calibration from IBM...")
        calibration = get_calibration_from_ibm(props, IBM_INSTANCE, IBM_TOKEN)
    except Exception as e:
        print(e)

coupling_edges = backend.configuration().coupling_map
coupling_edges = [tuple(sorted(edge)) for edge in coupling_edges]
coupling_map = CouplingMap(coupling_edges)


qc_qasm = QuantumCircuit.from_qasm_file("QASMfiles/"+qasmfile+".qasm")
qc = qc_qasm.decompose(gates_to_decompose=['ccx'])

#qc = qasm_to_qiskit(qasm_code)  
code_qiskit = circuit_to_qiskit_code(qc)
print("original Circuit Qiskit Code:")
print(code_qiskit)  

display(qc.draw('mpl', fold=-1))
qc_reordered = reorder_instructions(qc)
#display(qc_reordered.draw('mpl', fold=-1))
code_qiskit = circuit_to_qiskit_code(qc_reordered)
print("qc_reordered Circuit Qiskit Code:")
print(code_qiskit) 

original_qubit_count = qc_reordered.num_qubits 
qc_dyn = dynamic_circuit(qc_reordered)
qc_dyn =  compact_qubits(qc_dyn)
#display(qc_dyn.draw('mpl', fold=-1))
qc_dyn_qubit_count = qc_dyn.num_qubits 
print("Qubit Count:", qc_dyn_qubit_count)

code_qiskit = circuit_to_qiskit_code(qc_dyn)
print("Dynamic Circuit Qiskit Code:")
print(code_qiskit) 
#################create best initial mapping 
transpiled_qc_dyn = transpile(
    qc_dyn,
    backend,
    coupling_map = coupling_map,
    optimization_level=3
   # routing_method='sabre'      # Use SABRE for qubit mapping
)
display(plot_circuit_layout(transpiled_qc_dyn, backend))
# Get the initial layout
layout = transpiled_qc_dyn._layout.initial_layout
# Extract physical qubits of the main 'q' register
q_list = [
    phys for phys, qubit in layout.get_physical_bits().items()
    if f"QuantumRegister({qc_dyn_qubit_count}, 'q')" in str(qubit)
]
init_map = {i: q for i, q in enumerate(q_list)}  #{0: 0, 1: 1, 2: 2, 3: 3}

# Print the list
print("Logical -> Physical qubit mapping:", init_map)

################# qc_dyn  custom_coupling_edges

# Filter edges so both qubits are in q_list
custom_coupling_edges = [
    edge for edge in coupling_edges if edge[0] in q_list and edge[1] in q_list
]

# define coupling graph (physical topology)
cg = nx.Graph()
cg.add_edges_from(custom_coupling_edges)




# convert instructions to immutable nodes and build DAG
nodes = qiskit_instrs_to_immutable_nodes(qc_dyn)

#print(nodes)
#print("Original QC gates:", len(qc.data))
#print("Extracted DAG nodes:", len(nodes))
dag = build_dag_from_nodes(nodes)
#print("DAG nodes:", dag.number_of_nodes(), "edges:", dag.number_of_edges())
front = initialize_front_layer(dag)
dist = get_distance_matrix(cg)
# run beam-sabre


    
    
phys_to_idx = {v: k for k, v in init_map.items()}



router = BeamSabre(dist, cg)
prog_seq, final_map = router.execute_beam(
    front,
    init_map,
    phys_to_idx,
    dag,
    beam_width=4,
    alpha=1.0,
    beta=1.0 if calibration else 0.0,
    calibration=calibration,
)
'''for g in prog_seq:
    print(g)'''

# -------------------------
routed_qc = replay_to_qiskit(prog_seq, init_map)

code_qiskit = circuit_to_qiskit_code(routed_qc)
print("Dynamic Circuit Qiskit Code:")
print(code_qiskit) 

display(qc_dyn.draw('mpl', fold=-1))
display(routed_qc.draw('mpl', fold=-1))
#print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

#print("\nStats before:", count_stats(qc))
#print("Stats after :", count_stats(routed_qc))

def count_gates(qc: QuantumCircuit):
    """Return gate counts plus explicit CX, T, H, TDG stats."""
    stats = qc.count_ops()  # Counter object {gate: count}
    swap_count   = stats.get("swap", 0)
    cx_count   = stats.get("cx", 0)
    t_count    = stats.get("t", 0)
    h_count    = stats.get("h", 0)
    tdg_count  = stats.get("tdg", 0)

    print(f"Full stats: {dict(stats)}")
    print(f"SWAP  gates: {swap_count}")
    print(f"CX  gates: {cx_count}")
    print(f"T   gates: {t_count}")
    print(f"H   gates: {h_count}")
    print(f"TDG gates: {tdg_count}")
    return stats
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

# Example usage
print("\n Gates before:")
count_gates(qc)

print("\n Gates after:")
count_gates(routed_qc)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
######################################
#      RUN 
###################################
simulator = AerSimulator()
job = simulator.run(routed_qc, shots=shots)
result = job.result()
counts1 = result.get_counts(routed_qc)
#print(f"Measurement counts Simulator: {counts}")
for outcome, count in counts1.items():
    #print(f"{outcome}: {count}")
    file.write(f"SIM: {outcome}: {count} \n")
    
    
transpiled_qc = transpile(
    routed_qc,
    backend,
    coupling_map = coupling_map,
    #scheduling_method="asap",
    target=backend.target,
    optimization_level= 3,
    initial_layout = q_list
)


print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
'''duration_dt = transpiled_qc.duration
print("Duration (in dt units):", duration_dt)
dt = backend.configuration().dt  # seconds per dt
duration_sec = duration_dt * dt
print("Duration (in seconds):", duration_sec)'''

print("Standard depth:", transpiled_qc.depth())
print("Reset-aware depth:", reset_aware_depth(transpiled_qc))


print("Operations by type:", transpiled_qc.count_ops())
print("Total ops:", transpiled_qc.size())
print("Circuit depth:", transpiled_qc.depth())
print("Circuit 2Q gates depth:", transpiled_qc.depth(lambda x: x.operation.num_qubits == 2))
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")


display(plot_circuit_layout(transpiled_qc, backend))

'''sampler = SamplerV2(backend)  # or backend_real
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
    file.write(f"IBM: {outcome}: {count} \n")'''
file.close()

'''fidelity, mse = fidelity_from_two_counts(counts1, counts2)
print("Fidelity =", fidelity)
print("MSE Loss =", mse)'''