from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from collections import defaultdict, deque
from typing import Dict, List, Tuple
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# Function to move single-qubit gates backward after last CX or single-qubit gate
def reorder_instructions(qc: QuantumCircuit) -> QuantumCircuit:
    # Create a copy of the circuit instructions to avoid modifying the original
    # Each element: (gate instruction, qubit arguments, classical bit arguments)
    data = [(inst.copy(), list(qargs), list(cargs)) for inst, qargs, cargs in qc.data]

    # Initialize a list to track the last index where each qubit was involved in a gate
    last_gate_index = [-1] * qc.num_qubits

    i = 0  # Index to iterate over the instruction list

    # Iterate over all instructions
    while i < len(data):
        inst, qargs, cargs = data[i]  # Current instruction and its qubits/classical bits
        qubits = [qc.qubits.index(q) for q in qargs]  # Convert qubit objects to indices

        if len(qubits) == 1:  # Check if this is a single-qubit gate
            q = qubits[0]  # Index of the single qubit
            last_idx = last_gate_index[q]  # Last occurrence of a gate on this qubit

            # If the last gate on this qubit is not immediately before current, slide backward
            if last_idx != -1 and last_idx < i - 1:
                gate = data.pop(i)  # Remove current gate from its position
                data.insert(last_idx + 1, gate)  # Insert it just after the last gate on this qubit

                # Update last_gate_index for all qubits whose last gate index is greater than last_idx
                last_gate_index = [
                    idx + 1 if idx > last_idx else idx for idx in last_gate_index
                ]

                last_gate_index[q] = last_idx + 1  # Update last gate index for this qubit
                i += 1  # Move to next instruction
            else:
                last_gate_index[q] = i  # Update last gate index for this qubit
                i += 1  # Move to next instruction

        elif inst.name == 'cx':  # Check if this is a CX (controlled-NOT) gate
            for q in qubits:  # Update last gate index for both control and target qubits
                last_gate_index[q] = i
            i += 1  # Move to next instruction

        else:
            i += 1  # For all other gates, just continue

    # Rebuild a new QuantumCircuit with the optimized instruction order
    qc_new = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    for inst, qargs, cargs in data:
        qc_new.append(inst, qargs, cargs)  # Append each instruction in its new order

    return qc_new  # Return the optimized circuit


# Apply the reordering
def dynamic_circuit(original_qc: QuantumCircuit) -> QuantumCircuit:
    """
    Optimize a quantum circuit by dynamically reusing qubits.
    Adds measurement and reset before qubit reuse.

    If a single-qubit gate is the last operation on that qubit,
    it is applied immediately instead of being postponed.

    Args:
        original_qc (QuantumCircuit): Input circuit.

    Returns:
        QuantumCircuit: Optimized circuit with qubit reuse and measurement/reset.
    """

    # Step 1: Build a map of qubit usage: which instruction indices use each qubit
    usage_map: Dict[int, List[int]] = defaultdict(list)
    for idx, instruction in enumerate(original_qc.data):
        for q in instruction.qubits:
            usage_map[original_qc.qubits.index(q)].append(idx)

    # Step 2: Count total operations per qubit
    total_ops = {q: len(ops) for q, ops in usage_map.items()}

    # Step 3: Create new Quantum and Classical Registers
    new_qreg = QuantumRegister(original_qc.num_qubits, 'q')
    new_creg = ClassicalRegister(original_qc.num_clbits, 'c')
    optimized_qc = QuantumCircuit(new_qreg, new_creg)

    # Step 4: Initialize mapping and bookkeeping structures
    qubit_map: Dict[int, int] = {}            # Map logical qubits → physical qubits
    freed: deque = deque()                    # Queue of freed physical qubits
    postponed: Dict[int, List[Tuple]] = defaultdict(list)  # Postponed single-qubit gates
    applied_count: Dict[int, int] = defaultdict(int)       # Number of applied gates per qubit

    # Map each qubit to a classical bit for measurement
    measure_map: Dict[int, int] = {i: i for i in range(original_qc.num_qubits)}

    # Step 5: Iterate over all instructions in the original circuit
    for idx, instruction in enumerate(original_qc.data):
        instr = instruction.operation          # The gate operation
        qargs = instruction.qubits             # Qubits it acts on
        cargs = instruction.clbits             # Classical bits involved
        old_idxs = [original_qc.qubits.index(q) for q in qargs]  # Logical qubit indices

        # Step 5a: Handle single-qubit gates
        if len(qargs) == 1:
            old_idx = old_idxs[0]

            # If this is the last operation on this qubit
            if applied_count[old_idx] + 1 == total_ops[old_idx]:
                # Assign a physical qubit if not already mapped
                if old_idx not in qubit_map:
                    if freed:  # Reuse a freed qubit
                        new_idx = freed.popleft()
                        classical_bit = measure_map.get(new_idx, new_idx)
                        # Measure and reset the qubit before reuse
                        optimized_qc.measure(new_qreg[new_idx], new_creg[classical_bit])
                        optimized_qc.reset(new_qreg[new_idx])
                    else:  # Allocate a new qubit
                        new_idx = len(qubit_map)
                    qubit_map[old_idx] = new_idx

                # Apply the single-qubit gate to the mapped qubit
                optimized_qc.append(instr, [new_qreg[qubit_map[old_idx]]], cargs)
                applied_count[old_idx] += 1
                # Mark qubit as freed after applying last operation
                freed.append(qubit_map[old_idx])
            else:
                # Postpone gate for later application
                postponed[old_idx].append((instr, cargs))

        # Step 5b: Handle multi-qubit gates
        else:
            for old_idx in old_idxs:
                if old_idx not in qubit_map:
                    if freed:  # Reuse a freed qubit
                        new_idx = freed.popleft()
                        classical_bit = measure_map.get(new_idx, new_idx)
                        optimized_qc.measure(new_qreg[new_idx], new_creg[classical_bit])
                        optimized_qc.reset(new_qreg[new_idx])
                    else:  # Allocate a new qubit
                        new_idx = len(qubit_map)
                    qubit_map[old_idx] = new_idx

                # Apply any postponed single-qubit gates for this qubit
                for pend_instr, pend_cargs in postponed[old_idx]:
                    optimized_qc.append(
                        pend_instr, [new_qreg[qubit_map[old_idx]]], pend_cargs
                    )
                    applied_count[old_idx] += 1
                    if applied_count[old_idx] == total_ops.get(old_idx, 0):
                        # Free qubit if all gates applied
                        freed.append(qubit_map[old_idx])
                postponed[old_idx].clear()

            # Apply the multi-qubit gate to mapped qubits
            current_mapping = [new_qreg[qubit_map[old_idx]] for old_idx in old_idxs]
            optimized_qc.append(instr, current_mapping, cargs)

            # Update applied count and free qubits if done
            for old_idx in old_idxs:
                applied_count[old_idx] += 1
                if applied_count[old_idx] == total_ops.get(old_idx, 0):
                    freed.append(qubit_map[old_idx])

    # Step 6: Apply any remaining postponed single-qubit gates at the end
    for old_idx in list(postponed.keys()):
        if postponed[old_idx]:
            if old_idx not in qubit_map:
                if freed:
                    new_idx = freed.popleft()
                    classical_bit = measure_map.get(new_idx, new_idx)
                    optimized_qc.measure(new_qreg[new_idx], new_creg[classical_bit])
                    optimized_qc.reset(new_qreg[new_idx])
                else:
                    new_idx = len(qubit_map)
                qubit_map[old_idx] = new_idx

            for pend_instr, pend_cargs in postponed[old_idx]:
                optimized_qc.append(
                    pend_instr, [new_qreg[qubit_map[old_idx]]], pend_cargs
                )
                applied_count[old_idx] += 1
                if applied_count[old_idx] == total_ops.get(old_idx, 0):
                    freed.append(qubit_map[old_idx])
            postponed[old_idx].clear()

    # Step 7: Return the fully optimized circuit with qubit reuse
    return optimized_qc