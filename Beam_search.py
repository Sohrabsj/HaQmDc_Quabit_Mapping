#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 17:02:51 2025

@author: sohrab
"""

from typing import List, Tuple, Dict, Any, Optional
from heapq import nsmallest
import networkx as nx
from networkx import floyd_warshall_numpy
import numpy as np
from qiskit import QuantumCircuit

# -------------------------
def qiskit_instrs_to_immutable_nodes(qc: QuantumCircuit) -> List[Tuple[str, int, Tuple[int, ...]]]:
    nodes: List[Tuple[str, int, Tuple[int, ...]]] = []
    
    for idx, ci in enumerate(qc.data):
        inst = ci.operation
        qargs = ci.qubits
        # convert qargs (Qubit objects) to logical integer indices
        q_inds = tuple(qc.qubits.index(q) for q in qargs)
        inst_name = inst.name.lower()
        
        # Handle parameters for rotation gates
        if inst_name in {"rx", "ry", "rz"} and inst.params:
            param_str = "(" + ", ".join(f"{p}" for p in inst.params) + ",)"
        else:
            param_str = "()"  # empty for non-rotation gates

        inst_key = f"{inst_name}:{param_str}"
        nodes.append((inst_key, idx, q_inds))
    
    return nodes

# -------------------------
def build_dag_from_nodes(nodes: List[Tuple[str, int, Tuple[int, ...]]]) -> nx.DiGraph:
    """
    Builds a DAG (networkx.DiGraph) where nodes are the immutable node tuples (inst_key,index,qtuple).
    Add an edge u->v if v depends on u (they share any logical qubit and u.index < v.index).
    """
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n)
    # naive O(n^2) dependence detection (fine for small circuits; can optimize if needed)
    for i, u in enumerate(nodes):
        uq = set(u[2])
        for v in nodes[i + 1 :]:
            vq = set(v[2])
            if uq & vq:
                # u occurs before v and they share a qubit -> dependency
                G.add_edge(u, v)
    return G

# -------------------------
def initialize_front_layer(circuit_dag: nx.DiGraph) -> List[Tuple[str, int, Tuple[int, ...]]]:
    front = [n for n in circuit_dag.nodes() if circuit_dag.in_degree(n) == 0]
    return front


# -------------------------
def get_distance_matrix(coupling_graph: nx.Graph) -> np.ndarray:
    return floyd_warshall_numpy(coupling_graph)


# -------------------------
def calculate_distance(node, distance_matrix, mapping, phys_to_idx):
    if len(node[2]) < 2:
        return 0.0
    q0, q1 = node[2]
    p0, p1 = mapping[q0], mapping[q1]
    idx0, idx1 = phys_to_idx[p0], phys_to_idx[p1]
    return float(distance_matrix[idx0, idx1])


def create_extended_successor_set(F: List[Tuple[Any]], circuit_dag: nx.DiGraph, limit: int = 20) -> List[Tuple[Any]]:
    """
    Return up to `limit` unique successors of nodes in F.
    Deterministic (iteration order) and avoids duplicates.
    """
    E: List[Tuple[Any]] = []
    seen = set()
    for gate in F:
        for succ in circuit_dag.successors(gate):
            if succ in seen:
                continue
            
            seen.add(succ)
            E.append(succ)
            if len(E) >= limit:
                return E
    return E




def heuristic_function(F: List[Tuple], circuit_dag: nx.DiGraph, mapping: dict, distance_matrix: np.ndarray, swap_pair: Tuple[int, int], decay_parameter: List[float], phys_to_idx) -> float:
    E = create_extended_successor_set(F, circuit_dag, limit=20)
    size_E = len(E) if len(E) > 0 else 1
    size_F = len(F) if len(F) > 0 else 1
    W = 0.5  # weight
    a, b = swap_pair
    max_decay = max(decay_parameter[a] if a < len(decay_parameter) else 1.0,
                    decay_parameter[b] if b < len(decay_parameter) else 1.0)
    f_distance = sum(calculate_distance(g, distance_matrix, mapping, phys_to_idx) for g in F) / size_F
    e_distance = W * (sum(calculate_distance(g, distance_matrix, mapping, phys_to_idx) for g in E) / size_E)
    return max_decay * (f_distance + e_distance)

# -------------------------
def swap_hardware_cost(swap_pair: Tuple[int, int], mapping: dict, calibration: Optional[dict]) -> float:
    if not calibration:
        return 0.0
    l0, l1 = swap_pair
    p0, p1 = mapping[l0], mapping[l1]
    # edges keyed by physical-qubit tuples
    ekey = (p0, p1) if (p0, p1) in calibration.get("edges", {}) else (p1, p0)
    edge_info = calibration.get("edges", {}).get(ekey, {})
    cx_err = edge_info.get("cx_error", edge_info.get("error", 0.05))
    cx_time = edge_info.get("cx_gate_time", edge_info.get("gate_time", 300))
    q0 = calibration.get("qubits", {}).get(p0, {})
    q1 = calibration.get("qubits", {}).get(p1, {})
    ro0 = q0.get("readout_error", 0.02)
    ro1 = q1.get("readout_error", 0.02)
    cost = 100.0 * cx_err + 0.001 * cx_time + 10.0 * (ro0 + ro1) / 2.0
    t1_0 = q0.get("T1", 0.0)
    t1_1 = q1.get("T1", 0.0)
    if t1_0 > 0 and t1_1 > 0:
        avg_t1_us = ((t1_0 + t1_1) / 2.0) * 1e6
        cost += max(0.0, 10.0 - avg_t1_us / 10.0)

    return float(cost)


# -------------------------
EPSILON = 1e-2  # require at least this much improvement to accept a SWAP

def _ordered_pair(a: int, b: int) -> Tuple[int, int]:
    """Normalize a pair so (a,b) and (b,a) are treated as the same candidate."""
    return (a, b) if a <= b else (b, a)

def _last_swap(prog_seq: List[Tuple[str, Tuple[int, int]]]) -> Optional[Tuple[int, int]]:
    """Return last SWAP as an ordered pair, or None."""
    '''print(prog_seq)
    for op, pair in reversed(prog_seq):
        if op == "SWAP":
            return _ordered_pair(*pair)'''
        
    for instr in reversed(prog_seq):
        op = instr[0]
        pair = instr[1] if len(instr) > 1 else None
        if op == "SWAP":
            return _ordered_pair(*pair)
    return None


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4
class BeamSabre:
    def __init__(self, distance_matrix: np.ndarray, coupling_graph: nx.Graph):
        self.distance_matrix = distance_matrix
        self.coupling_graph = coupling_graph

    def initialize_decay_parameter(self, mapping: dict) -> List[float]:
        max_logical = max(mapping.keys()) if mapping else 0
        return [0.001] * (max_logical + 1)

    def is_executable(self, node: Tuple, mapping: dict) -> bool:
        if len(node[2]) < 2:
            return True
        q0, q1 = node[2]
        p0, p1 = mapping[q0], mapping[q1]
        return self.coupling_graph.has_edge(p0, p1)

    def get_logical_neighbours(self, logical: int, mapping: dict) -> List[int]:
        phys = mapping[logical]
        phys_nei = list(self.coupling_graph.neighbors(phys))
        logical_nei: List[int] = []
        phys_list = list(mapping.values())
        log_list = list(mapping.keys())
        for p in phys_nei:
            if p in phys_list:
                logical_nei.append(log_list[phys_list.index(p)])
        return logical_nei

    def update_mapping_swap(self, mapping: dict, swap_pair: Tuple[int, int]) -> dict:
        new_map = mapping.copy()
        a, b = swap_pair
        new_map[a], new_map[b] = mapping[b], mapping[a]
        return new_map

    def update_decay(self, decay: List[float], swap_pair: Tuple[int, int]) -> List[float]:
        new = decay[:]
        a, b = swap_pair
        if max(a, b) < len(new):
            new[a] += 0.001
            new[b] += 0.001
        return new

    def execute_beam(self,
                     front_layer: List[Tuple],
                     mapping: dict,
                     phys_to_idx: dict,
                     circuit_dag: nx.DiGraph,
                     beam_width: int = 4,
                     alpha: float = 1.0,
                     beta: float = 0.0,
                     calibration: Optional[dict] = None) -> Tuple[List[Tuple], dict]:
    
        total_nodes = set(circuit_dag.nodes())
        decay0 = self.initialize_decay_parameter(mapping)
        initial_state = (0.0, list(front_layer), mapping.copy(), [], decay0, set())  # (score, F, mapping, prog_seq, decay, done)
        beam = [initial_state]
    
        SWAP_PENALTY = 1e6   # very large penalty so penalized swaps are chosen only if necessary
    
        while beam:
            # If any beam state has executed all nodes, return it immediately (flush first)
            for s in beam:
                if len(s[5]) == len(total_nodes):
                    F_copy = list(s[1])
                    prog_copy = list(s[3])
                    done_copy = set(s[5])
                    flush_executable(F_copy, s[2], circuit_dag, prog_copy, done_copy, self)
                    return prog_copy, s[2]
    
            candidates = []
    
            for score, F, m, prog_seq, decay, done in beam:
                # === Execute executable gates in front layer ===
                exe = [n for n in F if self.is_executable(n, m)]
                if exe:
                    new_prog = list(prog_seq)
                    new_F = list(F)
                    new_m = m.copy()
                    new_decay = decay[:]
                    new_done = set(done)
    
                    for n in exe:
                        new_done.add(n)
                        gate_name = n[0].split(":")[0].upper()
                        qargs = n[2]
    
                        # Handle rotation gates with parameters
                        if gate_name in {"RX", "RY", "RZ"}:
                            param_str = n[0].split(":")[1]  # e.g., '(2.356,)'
                            param = float(param_str.strip("(),"))
                            new_prog.append((gate_name, param, qargs))
                        elif gate_name == "CNOT":
                            new_prog.append(("CNOT", qargs))
                        else:
                            new_prog.append((gate_name, qargs))
    
                        try:
                            new_F.remove(n)
                        except ValueError:
                            pass
    
                    # Add newly-ready successors
                    for n in exe:
                        for succ in circuit_dag.successors(n):
                            if succ in new_done:
                                continue
                            preds = list(circuit_dag.predecessors(succ))
                            if all(p in new_done for p in preds):
                                if succ not in new_F:
                                    new_F.append(succ)
    
                    candidates.append((score, new_F, new_m, new_prog, new_decay, new_done))
                    continue
    
                # === No executable gates: propose SWAPs ===
                seen_pairs: set[Tuple[int, int]] = set()
                swap_candidates: List[Tuple[int, int]] = []
                last_sw = _last_swap(prog_seq)  # avoid ping-pong
    
                for n in F:
                    if len(n[2]) < 2:
                        continue
                    q0, q1 = n[2]
                    c_nei = self.get_logical_neighbours(q0, m)
                    t_nei = self.get_logical_neighbours(q1, m)
    
                    for ln in c_nei:
                        if ln == q0:
                            continue
                        cand = _ordered_pair(q0, ln)
                        if cand not in seen_pairs and cand != last_sw:
                            seen_pairs.add(cand)
                            swap_candidates.append((q0, ln))
    
                    for ln in t_nei:
                        if ln == q1:
                            continue
                        cand = _ordered_pair(q1, ln)
                        if cand not in seen_pairs and cand != last_sw:
                            seen_pairs.add(cand)
                            swap_candidates.append((q1, ln))
    
                for sw in swap_candidates:
                    # Heuristic before and after swap
                    h0 = heuristic_function(F, circuit_dag, m, self.distance_matrix, sw, decay, phys_to_idx)
                    tmp_map = self.update_mapping_swap(m, sw)
                    h1 = heuristic_function(F, circuit_dag, tmp_map, self.distance_matrix, sw, decay, phys_to_idx)
                    hw = swap_hardware_cost(sw, m, calibration)
    
                    if h1 >= h0 - EPSILON:
                        comb = alpha * h1 + beta * hw + SWAP_PENALTY
                    else:
                        comb = alpha * h1 + beta * hw
    
                    new_prog = list(prog_seq)
                    new_prog.append(("SWAP", sw))
                    new_decay = self.update_decay(decay, sw)
                    candidates.append((score + comb, list(F), tmp_map, new_prog, new_decay, done.copy()))
    
            # If no candidates, attempt fallback swaps using physical edges
            if not candidates:
                fallback_candidates = []
                phys_list = list(mapping.values())
                log_list = list(mapping.keys())
                for (p, q) in sorted(self.coupling_graph.edges()):
                    if p in phys_list and q in phys_list:
                        la = log_list[phys_list.index(p)]
                        lb = log_list[phys_list.index(q)]
                        sw = _ordered_pair(la, lb)
                        rep_score, rep_F, rep_m, rep_prog, rep_decay, rep_done = beam[0]
                        h0 = heuristic_function(rep_F, circuit_dag, rep_m, self.distance_matrix, sw, rep_decay, phys_to_idx)
                        tmp_map = self.update_mapping_swap(rep_m, sw)
                        h1 = heuristic_function(rep_F, circuit_dag, tmp_map, self.distance_matrix, sw, rep_decay, phys_to_idx)
                        hw = swap_hardware_cost(sw, rep_m, calibration)
                        comb = alpha * h1 + beta * hw + SWAP_PENALTY
                        new_prog = list(rep_prog)
                        new_prog.append(("SWAP", sw))
                        new_decay = self.update_decay(rep_decay, sw)
                        fallback_candidates.append((rep_score + comb, list(rep_F), tmp_map, new_prog, new_decay, rep_done.copy()))
                if fallback_candidates:
                    candidates.extend(nsmallest(beam_width, fallback_candidates, key=lambda x: x[0]))
    
            if not candidates:
                best = min(beam, key=lambda x: x[0])
                F_copy = list(best[1])
                prog_copy = list(best[3])
                done_copy = set(best[5])
                flush_executable(F_copy, best[2], circuit_dag, prog_copy, done_copy, self)
                return prog_copy, best[2]
    
            # Keep top-K by score
            beam = nsmallest(beam_width, candidates, key=lambda x: x[0])
    
            # Merge duplicates by (mapping, front signature, done); keep best score
            merged: Dict[Any, Any] = {}
            for s in beam:
                mapping_key = tuple(sorted(s[2].items()))
                fl_key = tuple(sorted([(tuple(n[2]) + (n[1],)) for n in s[1]]))
                key = (mapping_key, fl_key, tuple(sorted(s[5])))
                if key not in merged or s[0] < merged[key][0]:
                    merged[key] = s
            beam = list(merged.values())
            beam = nsmallest(beam_width, beam, key=lambda x: x[0])
    
        # Final fallback
        best = min(beam, key=lambda x: x[0])
        F_copy = list(best[1])
        prog_copy = list(best[3])
        done_copy = set(best[5])
        flush_executable(F_copy, best[2], circuit_dag, prog_copy, done_copy, self)
        return prog_copy, best[2]

# -------------------------
def get_calibration_from_ibm(props, instance: Optional[str], token: Optional[str]) -> dict:

    calib = {"edges": {}, "qubits": {}}
    # qubits
    for qidx, qprops in enumerate(props.qubits):
        ro = None
        T1 = None
        T2 = None
        for p in qprops:
            nm = p.name.lower() if hasattr(p, 'name') else ''
            if nm.startswith('Readout assignment') or nm.startswith('readout_error'):
                try:
                    ro = float(p.value)
                except Exception:
                    ro = None
            if nm == 't1':
                T1 = p.value
            if nm == 't2':
                T2 = p.value
        calib['qubits'][qidx] = {'readout_error': float(ro) if ro is not None else 0.02,
                                 'T1': float(T1) if T1 is not None else 0.0,
                                 'T2': float(T2) if T2 is not None else 0.0}
    # gates
    for g in props.gates:
        gate_name = getattr(g, 'gate', None)
        if gate_name and gate_name.lower() in ('cx', 'cnot', 'ecr', 'ECR error'):
            qubits = tuple(g.qubits)
            err = None
            length = None
            for p in g.parameters:
                nm = p.name.lower()
                if 'error' in nm:
                    err = p.value
                if 'gate_length' in nm:
                    length = p.value
            calib['edges'][qubits] = {'cx_error': float(err) if err is not None else 0.05,
                                      'cx_gate_time': float(length) if length is not None else 300}
    return calib


def count_stats(qc: QuantumCircuit) -> Dict[str, int]:
    cx = 0
    swap = 0
    # Use CircuitInstruction API explicitly
    for ci in qc.data:
        op = ci.operation
        nm = getattr(op, 'name', '').lower()
        if nm in ('cx', 'cnot', 'ecr'):
            cx += 1
        elif nm == 'swap':
            swap += 1
    return {"cx": cx, "swap": swap, "effective_cx": cx + 3 * swap, "depth": qc.depth()}


def flush_executable(F: List[Tuple], mapping: dict, circuit_dag: nx.DiGraph,
                     prog_seq: List[Tuple], done: set, beam_obj: 'BeamSabre') -> None:
    """
    Greedily execute any gates in F that are executable under mapping.
    Update prog_seq, F (in-place), and done (in-place).

    This is used right before returning a partial solution to avoid
    missing operations that are already executable.
    """
    
    made_progress = True

    while made_progress:
        made_progress = False
        for n in list(F):
            print("nnnnnnnn", n)
            if beam_obj.is_executable(n, mapping):
                made_progress = True
                done.add(n)

                # --- Two-qubit gate ---
                if len(n[2]) >= 2:
                    q0, q1 = n[2]
                    prog_seq.append(("CNOT", (q0, q1)))

                # --- Single-qubit gate ---
                else:
                    q0 = n[2][0]
                    gate_name = n[0].split(":")[0].upper()

                    # --- Handle rotation gates with parameters ---
                    if gate_name in ("RX", "RY", "RZ"):
                        # extract parameter from n[0]
                        # n[0] format: "rx:(2.356,)" or "rz:(1.571,)"
                        try:
                            param_str = n[0].split(":")[1].strip("()")
                            param = float(param_str)
                        except Exception:
                            param = 0.0  # fallback
                        prog_seq.append((gate_name, param, (q0,)))

                    # --- Other single-qubit gates ---
                    else:
                        prog_seq.append((gate_name, (q0,)))

                # Remove from front layer
                try:
                    F.remove(n)
                except ValueError:
                    pass

                # Add newly-ready successors
                for succ in circuit_dag.successors(n):
                    if succ in done:
                        continue
                    preds = list(circuit_dag.predecessors(succ))
                    if all(p in done for p in preds) and succ not in F:
                        F.append(succ)

                # break to re-evaluate F from the top (keeps determinism)
                break


