from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

def qasm_to_qiskitbbbbb(qasm_code: str) -> QuantumCircuit:
    """
    Convert QASM code (as string) to a Qiskit QuantumCircuit,
    preserving all declared qubits and clbits.
    """
    try:
        qc = QuantumCircuit.from_qasm_str(qasm_code)  # preserves full registers
        return qc
    except Exception as e:
        print("Error converting QASM to Qiskit:", e)
        return None
from qiskit import QuantumCircuit, transpile

from qiskit import QuantumCircuit

def qasm_to_qiskit(qasm_code: str) -> QuantumCircuit:
    """
    Convert QASM code (as string) to a Qiskit QuantumCircuit,
    preserving all declared qubits and clbits.
    Only decompose CCX (Toffoli) gates into {cx, h, t, tdg}.
    All other gates remain unchanged.
    """
    try:
        qc = QuantumCircuit.from_qasm_str(qasm_code)

        # Decompose only CCX gates, leave others intact
        decomposed_qc = qc.decompose(gates_to_decompose=['ccx'])

        return decomposed_qc

    except Exception as e:
        print("Error converting QASM to Qiskit:", e)
        return None
def circuit_to_qiskit_code(qc_cir: QuantumCircuit) -> str:
    """
    Convert a QuantumCircuit object to Python Qiskit code.
    Returns a string that can be copied and executed.
    """
    lines = []
    n_qubits = qc_cir.num_qubits
    print(n_qubits)
    n_clbits = qc_cir.num_clbits
    lines.append(f"from qiskit import QuantumCircuit")
    lines.append(f"qc = QuantumCircuit({n_qubits}, {n_clbits})\n")

    for inst, qargs, cargs in qc_cir.data:
        # Get qubit indices
        q_indices = [qc_cir.qubits.index(q) for q in qargs]
        # Get clbit indices
        c_indices = [qc_cir.clbits.index(c) for c in cargs]

        # Prepare qubit and clbit strings
        q_str = ", ".join([f"{q}" for q in q_indices])
        c_str = ", ".join([f"{c}" for c in c_indices])

        # Build gate string
        if inst.params:  # parameterized gate
            params = ", ".join([str(p) for p in inst.params])
            if c_indices:
                lines.append(f"qc.{inst.name}({params}, [{q_str}], [{c_str}])")
            else:
                lines.append(f"qc.{inst.name}({params}, {q_str})")
        else:  # non-parameterized gate
            if c_indices:
                lines.append(f"qc.{inst.name}([{q_str}], [{c_str}])")
            else:
                lines.append(f"qc.{inst.name}({q_str})")

    return "\n".join(lines)
def compact_qubits(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Remap qubits in the circuit to a compact range (q[0], q[1], ..., q[n-1])
    by removing unused qubits and updating gate references accordingly.
    Classical bits are preserved as-is.
    """
    # Compute used qubit indices
    used_qubits = sorted({qc.qubits.index(q) for instr in qc.data for q in instr.qubits})
    qubit_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_qubits)}

    # Create new compact quantum register
    new_qreg = QuantumRegister(len(used_qubits), 'q')
    new_creg = ClassicalRegister(len(qc.clbits), 'c')  # preserve all clbits
    new_qc = QuantumCircuit(new_qreg, new_creg)

    # Remap each instruction
    for instruction in qc.data:
        instr = instruction.operation
        qargs = instruction.qubits
        cargs = instruction.clbits
        mapped_qargs = [new_qreg[qubit_map[qc.qubits.index(q)]] for q in qargs]
        mapped_cargs = [new_creg[qc.clbits.index(c)] for c in cargs]
        new_qc.append(instr, mapped_qargs, mapped_cargs)

    return new_qc

from typing import List, Tuple, Dict
from qiskit import QuantumCircuit, ClassicalRegister
'''
def update_prog_seq_after_swap(prog_seq: list[tuple[str, tuple[int, ...]]],
                               swap_qubits: tuple[int, int],
                               start_idx: int = 0) -> None:
    """
    Update prog_seq in-place after a SWAP between qubits a and b.
    All operations after start_idx that reference a or b are updated.

    Args:
        prog_seq: list of (gate_name, qubits) tuples
        swap_qubits: tuple (a, b) of logical qubits that were swapped
        start_idx: index in prog_seq to start updating from
    """
    a, b = swap_qubits
    for i in range(start_idx, len(prog_seq)):
        op, qubits = prog_seq[i]

        # Support single- and multi-qubit gates
        if isinstance(qubits, int):
            if qubits == a:
                prog_seq[i] = (op, b)
            elif qubits == b:
                prog_seq[i] = (op, a)
        else:
            new_qubits = tuple(b if q == a else a if q == b else q for q in qubits)
            prog_seq[i] = (op, new_qubits)

from qiskit import QuantumCircuit, ClassicalRegister
from typing import List, Tuple, Dict

def replay_to_qiskit(prog_seq: List[Tuple], initial_mapping: Dict[int, int]) -> QuantumCircuit:
    """
    Recreate a QuantumCircuit from a program sequence using logical qubits,
    and measure all logical qubits at the end.

    Parameters:
        prog_seq: List of tuples representing operations. Examples:
                  ("H", (0,)), ("CX", (0,1)), ("RX", 2.356, (0,))
        initial_mapping: dict mapping logical -> physical qubits (used for SWAP tracking)

    Returns:
        QuantumCircuit using logical qubits.
    """
    if not initial_mapping:
        raise ValueError("initial_mapping must be provided")

    # Number of logical qubits
    n_logical = max(initial_mapping.keys()) + 1

    # Create circuit with logical qubits
    qc = QuantumCircuit(n_logical)

    # Count total measurements needed: explicit + final all-qubits
    n_explicit_meas = sum(1 for op in prog_seq if op[0].upper() == "MEASURE")
    total_cbits = n_explicit_meas + n_logical
    qc.add_register(ClassicalRegister(total_cbits))

    temp_map = initial_mapping.copy()  # logical -> current physical (for SWAP tracking)
    print(temp_map[0])
    cbit_index = 0

    # Replay gates
    for idx, op_tuple in enumerate(prog_seq):
        op_u = op_tuple[0].upper()

        # --- Handle SWAP separately ---
        if op_u == "SWAP":
            a, b = op_tuple[1]
            # Apply SWAP in circuit
            qc.swap(a, b)

        # --- Handle CNOT / CX ---
        elif op_u in ("CNOT", "CX"):
            qc.cx(*op_tuple[1])

        # --- Handle measurement ---
        elif op_u == "MEASURE":
            for a in op_tuple[1]:
                qc.measure(a, cbit_index)
                cbit_index += 1

        # --- Handle RESET ---
        elif op_u == "RESET":
            for a in op_tuple[1]:
                qc.reset(a)

        # --- Handle rotation gates with parameter ---
        elif op_u in ("RX", "RY", "RZ"):
            param, qubits = op_tuple[1], op_tuple[2]
            getattr(qc, op_u.lower())(param, qubits[0])

        # --- Handle other single-qubit gates ---
        else:
            qubits = op_tuple[1]
            getattr(qc, op_u.lower())(qubits[0])

    # --- Measure all remaining logical qubits that haven't been explicitly measured ---
    for a in range(n_logical):
        qc.measure(a, cbit_index)
        cbit_index += 1

    return qc
'''

from typing import List, Tuple, Dict
from qiskit import QuantumCircuit, ClassicalRegister

def update_prog_seq_after_swap(
    prog_seq: list[tuple],
    swap_qubits: tuple[int, int],
    start_idx: int = 0
) -> None:
    """
    Update prog_seq in-place after a SWAP between qubits a and b.
    All operations after start_idx that reference a or b are updated.

    Args:
        prog_seq: list of (gate_name, qubits) tuples
        swap_qubits: tuple (a, b) of logical qubits that were swapped
        start_idx: index in prog_seq to start updating from
    """
    a, b = swap_qubits
    for i in range(start_idx, len(prog_seq)):
        op = prog_seq[i][0]
        args = prog_seq[i][1:]

        # Replace inside qubit tuples/ints
        new_args = []
        for arg in args:
            if isinstance(arg, int):
                if arg == a:
                    new_args.append(b)
                elif arg == b:
                    new_args.append(a)
                else:
                    new_args.append(arg)
            elif isinstance(arg, tuple):
                new_args.append(
                    tuple(
                        b if q == a else a if q == b else q
                        for q in arg
                    )
                )
            else:
                new_args.append(arg)

        prog_seq[i] = (op, *new_args)


def replay_to_qiskit(
    prog_seq: List[Tuple],
    initial_mapping: Dict[int, int]
) -> QuantumCircuit:
    """
    Recreate a QuantumCircuit from a program sequence using logical qubits,
    and measure all logical qubits at the end.

    Parameters:
        prog_seq: List of tuples representing operations. Examples:
                  ("H", (0,)), ("CX", (0,1)), ("RX", 2.356, (0,))
        initial_mapping: dict mapping logical -> physical qubits (unused here but
                         can be kept for consistency)

    Returns:
        QuantumCircuit using logical qubits.
    """
    if not initial_mapping:
        raise ValueError("initial_mapping must be provided")

    n_logical = max(initial_mapping.keys()) + 1
    qc = QuantumCircuit(n_logical)

    # Add classical register for explicit + final measurements
    n_explicit_meas = sum(1 for op in prog_seq if op[0].upper() == "MEASURE")
    total_cbits = n_explicit_meas + n_logical
    qc.add_register(ClassicalRegister(total_cbits))

    cbit_index = 0
    i = 0
    while i < len(prog_seq):
        op = prog_seq[i][0].upper()

        if op == "SWAP":
            a, b = prog_seq[i][1]
            qc.swap(a, b)
            # Update all subsequent instructions
            update_prog_seq_after_swap(prog_seq, (a, b), i + 1)

        elif op in ("CNOT", "CX"):
            qc.cx(*prog_seq[i][1])

        elif op == "MEASURE":
            for a in prog_seq[i][1]:
                qc.measure(a, cbit_index)
                cbit_index += 1

        elif op == "RESET":
            for a in prog_seq[i][1]:
                qc.reset(a)

        elif op in ("RX", "RY", "RZ"):
            param, qubits = prog_seq[i][1], prog_seq[i][2]
            getattr(qc, op.lower())(param, qubits[0])

        else:
            qubits = prog_seq[i][1]
            getattr(qc, op.lower())(qubits[0])

        i += 1

    # Final measure of all logical qubits
    for a in range(n_logical):
        qc.measure(a, cbit_index)
        cbit_index += 1

    return qc


from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

def reset_aware_depth(circuit: QuantumCircuit) -> int:
    """
    Compute circuit depth, treating 'reset' as restarting
    the depth counter for affected qubits.
    
    Parameters:
        circuit (QuantumCircuit): Input circuit.
    
    Returns:
        int: Reset-aware depth.
    """
    dag = circuit_to_dag(circuit)
    # Track depth for each qubit
    depths = {q: 0 for q in dag.qubits}
    max_depth = 0

    for node in dag.topological_op_nodes():
        qbs = node.qargs

        if node.name == "reset":
            # Count reset itself
            for qb in qbs:
                depths[qb] += 1
                max_depth = max(max_depth, depths[qb])
                # Restart timeline for that qubit
                depths[qb] = 0  
        else:
            # Layer = 1 + max depth of qubits involved
            layer = 1 + max(depths[qb] for qb in qbs)
            for qb in qbs:
                depths[qb] = layer
            max_depth = max(max_depth, layer)

    return max_depth
import numpy as np

'''def fidelity_from_two_counts(counts1, counts2):
    """
    Compute fidelity between two measurement results (counts).
    
    Args:
        counts1 (dict): First counts dictionary.
        counts2 (dict): Second counts dictionary.
    
    Returns:
        float: Fidelity value in [0, 1].
    """
    # Normalize counts to probabilities
    shots1 = sum(counts1.values())
    shots2 = sum(counts2.values())
    probs1 = {k: v / shots1 for k, v in counts1.items()}
    probs2 = {k: v / shots2 for k, v in counts2.items()}
    
    # All possible outcomes (union of keys)
    all_outcomes = set(probs1.keys()).union(probs2.keys())
    
    # Fidelity formula
    fidelity = (
        sum(np.sqrt(probs1.get(o, 0.0) * probs2.get(o, 0.0)) for o in all_outcomes)
    ) ** 2
    
    return fidelity
'''
import numpy as np

def fidelity_from_two_counts(counts1, counts2):
    """
    Compute fidelity and MSE loss between two measurement results (counts).
    
    Args:
        counts1 (dict): First counts dictionary.
        counts2 (dict): Second counts dictionary.
    
    Returns:
        tuple: (fidelity, mse_loss)
    """
    # Normalize counts to probabilities
    shots1 = sum(counts1.values())
    shots2 = sum(counts2.values())
    probs1 = {k: v / shots1 for k, v in counts1.items()}
    probs2 = {k: v / shots2 for k, v in counts2.items()}
    
    # All possible outcomes (union of keys)
    all_outcomes = set(probs1.keys()).union(probs2.keys())
    
    # Fidelity
    fidelity = (
        sum(np.sqrt(probs1.get(o, 0.0) * probs2.get(o, 0.0)) for o in all_outcomes)
    ) ** 2
    
    # MSE loss
    mse_loss = np.mean([(probs1.get(o, 0.0) - probs2.get(o, 0.0)) ** 2 for o in all_outcomes])
    
    return fidelity, mse_loss


