import numpy as np
import random
from graph_tool.all import Graph
from graph_tool.topology import shortest_path
import math
import heapq
import matplotlib.pyplot as plt
import datetime
import os
import concurrent.futures
import time
import networkx as nx


def generate_connection_matrix(
    num_nodes,
    num_communities=2,
    intra_prob=0.4,         # additional ER random edges inside communities
    inter_prob=0.05,        # sparse edges between communities
    use_scale_free=True,    # use BA scale-free model inside communities
    use_small_world=True,   # small-world rewiring
    rewiring_p=0.1,         # rewiring probability
    ensure_end_connected=True
):
    """
    Generate a 0/1 adjacency matrix with realistic network properties.

    Realistic topology features:
      - Multi-community modular structure
      - Scale-free (with hubs) structure inside each community
      - Small-world rewiring
      - Backbone-style weak connections between communities
      - Ensure source=0 → target=N-1 are connected
    """

    G = nx.Graph()

    # ---------------- Step 1: Partition nodes into communities ----------------
    base_size = num_nodes // num_communities
    remainder = num_nodes % num_communities
    communities = []
    start = 0

    for c in range(num_communities):
        size_c = base_size + (1 if c < remainder else 0)
        end = start + size_c
        communities.append(list(range(start, end)))
        start = end

    # ---------------- Step 2: Internal structure of each community ----------------
    for nodes in communities:
        k = len(nodes)
        if k == 1:
            continue

        # 2.1 Scale-free (BA model)
        if use_scale_free and k >= 3:
            m = min(2, k - 1)
            G_sub = nx.barabasi_albert_graph(k, m)
        else:
            # fallback ER graph
            G_sub = nx.erdos_renyi_graph(k, intra_prob)

        # Relabel nodes inside this community to global node indices
        mapping = {i: nodes[i] for i in range(k)}
        G_sub = nx.relabel_nodes(G_sub, mapping)

        # Add into the global graph
        G.add_nodes_from(G_sub.nodes())
        G.add_edges_from(G_sub.edges())

        # 2.2 Add some extra random internal edges
        for i in range(k):
            for j in range(i + 1, k):
                if random.random() < intra_prob:
                    u, v = nodes[i], nodes[j]
                    G.add_edge(u, v)

    # ---------------- Step 3: Sparse backbone-style edges between communities ----------------
    for c1 in range(num_communities):
        for c2 in range(c1 + 1, num_communities):
            for u in communities[c1]:
                for v in communities[c2]:
                    if random.random() < inter_prob:
                        G.add_edge(u, v)

    # ---------------- Step 4: Small-world rewiring ----------------
    if use_small_world:
        edges = list(G.edges())
        for (u, v) in edges:
            if random.random() < rewiring_p:
                new_v = random.randint(0, num_nodes - 1)
                if new_v != u and not G.has_edge(u, new_v):
                    G.remove_edge(u, v)
                    G.add_edge(u, new_v)

    # ---- Step 5: Ensure connectivity from 0 to N-1 (with max trials + fallback direct edge) ----
    if ensure_end_connected and num_nodes >= 2:
        max_try = 1000     # you can tune this as needed
        tries = 0
        while (not nx.has_path(G, 0, num_nodes - 1)) and (tries < max_try):
            # Prefer to expand the component containing 0 via an intermediate node
            mid = random.randint(1, num_nodes - 2) if num_nodes > 2 else num_nodes - 1
            G.add_edge(0, mid)
            tries += 1

        # If still disconnected after max_try attempts, directly connect 0 and N-1
        if not nx.has_path(G, 0, num_nodes - 1):
            G.add_edge(0, num_nodes - 1)

    # ---------------- Step 6: Convert to adjacency matrix ----------------
    adj = np.zeros((num_nodes, num_nodes), dtype=int)
    for u, v in G.edges():
        adj[u, v] = 1
        adj[v, u] = 1

    return adj


def build_graphtool_graph_distance(topology_matrix):
    """
    Construct a graph-tool graph using only the topology matrix,
    and set weight = 1.0 for every existing edge.
    This is used for the traditional version that selects paths purely
    by hop-length distance.
    """
    N = topology_matrix.shape[0]
    g = Graph(directed=False)
    g.add_vertex(N)

    w = g.new_edge_property("double")  # weight

    for i in range(N):
        for j in range(i + 1, N):
            if topology_matrix[i, j] == 1:
                e = g.add_edge(i, j)
                w[e] = 1.0   # each edge length = 1

    return g, w


def select_paths_traditional_distance(topology_matrix, num_paths=3, max_path_len=None):
    """
    Traditional version path selection:
      - Does NOT use trust_score at all;
      - Uses only the topology (edge length = 1) to find the K shortest simple paths;
      - Directly takes the first num_paths paths without trust-based security ranking
        or correlation filtering.
    """
    N = topology_matrix.shape[0]
    source, target = 0, N - 1

    # Use the "pure distance graph"
    g, w = build_graphtool_graph_distance(topology_matrix)

    # Find num_paths shortest simple paths
    K_cand = num_paths   # could also use num_paths * 3 for more candidates; here we keep it simple
    candidate_paths = yen_k_shortest_paths_gt(
        g, w, source, target, K=K_cand, max_path_len=max_path_len
    )

    if not candidate_paths:
        return []

    # Directly take the first num_paths
    return candidate_paths[:num_paths]


# Eve choose nodes to attack
def generate_attack_matrix(num_nodes, AP):
    """
    Generate an attack-occupation matrix.

    Args:
        num_nodes (int): number of nodes
        AP (float): attack pervasiveness (0~1), fraction of nodes that are attacked

    Returns:
        np.ndarray: num_nodes × num_nodes matrix, A[i][i]=1 means node i is occupied.
    """
    M = np.zeros((num_nodes, num_nodes), dtype=int)
    num_attacked = int(num_nodes * AP)
    possible_nodes = list(range(1, num_nodes - 1))
    num_attacked = min(num_attacked, len(possible_nodes))
    attacked_nodes = random.sample(possible_nodes, num_attacked)
    for node in attacked_nodes:
        M[node, node] = 1
    return M


def generate_attack_matrices_for_AP_range(AP_min, AP_max, AP_step, num_nodes=10):
    """
    Generate a set of attack-occupation matrices (diagonal 1 means attacked node)
    corresponding to each AP value from AP_min to AP_max.

    Endpoints 0 and num_nodes-1 will never be attacked.

    Args:
        AP_min (float): starting AP value (inclusive)
        AP_max (float): ending AP value (inclusive)
        AP_step (float): AP step
        num_nodes (int): total number of nodes, default 10

    Returns:
        tuple:
            ap_list (np.ndarray): all AP values
            attack_matrices (np.ndarray): integer array of shape
                                          (len(ap_list), num_nodes, num_nodes)
    """

    if AP_step <= 0:
        raise ValueError("AP_step must be positive")

    # Generate AP list including AP_max (consider floating-point issues)
    ap_list = np.arange(AP_min, AP_max + 1e-12, AP_step)
    ap_list = np.round(ap_list, 12)

    M = len(ap_list)
    N = int(num_nodes)
    if N < 2:
        raise ValueError("num_nodes must be at least 2")

    attack_matrices = np.zeros((M, N, N), dtype=int)

    # Local function: generate a single attack matrix for a given AP
    def _generate_attack_matrix(N, AP):
        mat = np.zeros((N, N), dtype=int)
        k = int(round(AP * N))
        max_nodes = max(0, N - 2)  # exclude first and last node
        k = min(k, max_nodes)
        if k > 0:
            candidates = list(range(1, N - 1))
            selected = random.sample(candidates, k)
            for node in selected:
                mat[node, node] = 1
        return mat

    for i, ap in enumerate(ap_list):
        attack_matrices[i] = _generate_attack_matrix(N, ap)

    return ap_list, attack_matrices


# initialize trust_matrix
def initialize_trust_matrix(topology_matrix):
    """
    Initialize a trust matrix based on the topology matrix.
    Diagonal entries are trust values, set to 0.999, while off-diagonals
    follow the connectivity (0 or 1).

    Args:
        topology_matrix (np.ndarray): adjacency matrix, A[i,j]=1 means nodes i and j are connected

    Returns:
        np.ndarray: initial trust matrix, diagonal≈1, off-diagonal = topology (0 or 1)
    """
    trust_matrix = topology_matrix.copy().astype(float)
    N = trust_matrix.shape[0]

    # Set all diagonal trust values to 0.999
    for i in range(N):
        trust_matrix[i, i] = 0.999

    return trust_matrix


# Routing (core part of ZTMPQKD)
def build_graphtool_graph(trust_matrix):
    """
    Convert trust_matrix (0/1 edges + diagonal trust) into a graph-tool Graph and edge-weight property.

    Returns:
        (g, w)
    """
    N = trust_matrix.shape[0]
    g = Graph(directed=False)
    g.add_vertex(N)

    w = g.new_edge_property("double")  # weight

    for i in range(N):
        for j in range(i + 1, N):
            if trust_matrix[i, j] == 1:
                edge_trust = (trust_matrix[i, i] + trust_matrix[j, j]) / 2
                weight = -math.log(edge_trust + 1e-8)
                e = g.add_edge(i, j)
                w[e] = weight
    return g, w


def yen_k_shortest_paths_gt(g, w, source, target, K, max_path_len=None):
    """
    Yen's algorithm using graph-tool to find K shortest simple paths (no cycles).

    Args:
        g: graph-tool Graph (undirected)
        w: edge weight property
        source/target: int
        K: number of paths
        max_path_len: optional limit on number of nodes in a path; None means no limit

    Returns:
        list[list[int]]
    """

    def path_weight(path):
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            e = g.edge(u, v)
            if e is None:
                return float("inf")
            total += w[e]
        return total

    # --- 1) Find the shortest path P0 ---
    vlist, elist = shortest_path(
        g,
        source=g.vertex(source),
        target=g.vertex(target),
        weights=w
    )
    if len(vlist) == 0:
        return []
    P0 = [int(v) for v in vlist]
    if max_path_len is not None and len(P0) > max_path_len:
        return []

    A = [P0]  # confirmed shortest paths
    A_weights = [path_weight(P0)]

    # B is a min-heap of candidate paths: (cost, path)
    B = []

    # For deduplication (avoid pushing identical paths into the heap)
    seen = {tuple(P0)}

    # --- 2) Iteratively find P1..P_{K-1} ---
    for k in range(1, K):
        prev_path = A[k - 1]

        # Spur node index from 0 to len(prev_path)-2 (no need to spur the last node)
        for i_spur in range(len(prev_path) - 1):
            spur_node = prev_path[i_spur]
            root_path = prev_path[:i_spur + 1]

            # max_path_len pruning (if root is already too long)
            if max_path_len is not None and len(root_path) >= max_path_len:
                continue

            # ---- Temporarily remove the next edge of any path in A that shares this root prefix ----
            removed_edges = []

            for p in A:
                if len(p) > i_spur and p[:i_spur + 1] == root_path:
                    u = p[i_spur]
                    v = p[i_spur + 1]
                    e = g.edge(u, v)
                    if e is not None:
                        w_old = float(w[e])          # record original weight
                        g.remove_edge(e)
                        removed_edges.append((u, v, w_old))

            # (No vertex clearing here; we avoid cycles via a simple-path check below)

            # ---- Find shortest spur_path from spur_node to target ----
            vspur, espur = shortest_path(
                g,
                source=g.vertex(spur_node),
                target=g.vertex(target),
                weights=w
            )

            if len(vspur) > 0:
                spur_path = [int(v) for v in vspur]
                total_path = root_path[:-1] + spur_path  # join

                # Simple path check: repeated nodes imply a cycle → skip
                if len(set(total_path)) == len(total_path):
                    if max_path_len is None or len(total_path) <= max_path_len:
                        ttp = tuple(total_path)
                        if ttp not in seen:
                            seen.add(ttp)
                            cost = path_weight(total_path)
                            heapq.heappush(B, (cost, total_path))

            # ---- Restore the removed edges ----
            for u, v, w_old in removed_edges:
                e_restore = g.add_edge(u, v)
                w[e_restore] = w_old

        if not B:
            break

        # Pop the smallest-cost candidate path as the next shortest path
        cost_k, path_k = heapq.heappop(B)
        A.append(path_k)
        A_weights.append(cost_k)

    return A


def select_paths(trust_matrix, num_paths=3, lambda_cor=0.5, max_path_len=None):
    """
    Path selection using graph-tool + Yen K-shortest:

      1) Use Yen's algorithm to find K shortest simple paths (by weighted distance).
      2) Take num_paths*3 candidates.
      3) Sort by security and then greedily filter by correlation threshold.
    """
    N = trust_matrix.shape[0]
    source, target = 0, N - 1

    g, w = build_graphtool_graph(trust_matrix)

    # ---- 1) Yen K-shortest paths ----
    K_cand = num_paths * 3
    candidate_paths = yen_k_shortest_paths_gt(
        g, w, source, target, K=K_cand, max_path_len=max_path_len
    )

    if not candidate_paths:
        return []

    # ---- 2) Path security (same as original definition) ----
    def path_security(path):
        p = 1.0
        for node in path:
            p *= trust_matrix[node, node]
        return p

    sec_values = [path_security(p) for p in candidate_paths]

    # ---- 3) Path correlation ----
    def path_correlation(p1, p2):
        edges1 = set(tuple(sorted((p1[i], p1[i + 1]))) for i in range(len(p1) - 1))
        edges2 = set(tuple(sorted((p2[i], p2[i + 1]))) for i in range(len(p2) - 1))
        common = len(edges1 & edges2)
        total = len(edges1 | edges2)
        return common / total if total else 0.0

    cor_values = []
    for i in range(len(candidate_paths)):
        for j in range(i + 1, len(candidate_paths)):
            cor_values.append(path_correlation(candidate_paths[i], candidate_paths[j]))

    avg_cor = np.mean(cor_values) if cor_values else 0.0
    TH_Cor = lambda_cor * avg_cor

    # ---- 4) Greedy selection (sorted by security in descending order) ----
    selected = []
    for path, sec in sorted(zip(candidate_paths, sec_values), key=lambda x: -x[1]):
        if not selected:
            selected.append(path)
        else:
            corr_ok = all(path_correlation(path, p_sel) < TH_Cor for p_sel in selected)
            if corr_ok:
                selected.append(path)

        if len(selected) >= num_paths:
            break

    return selected


# single step of transmission (update trust matrix)
def simulate_time_step(trust_matrix, path_library, num_multipath,
                       attack_matrix, beta_r, beta_p, gama_t):
    """
    Simulate one time step of multi-path quantum transmission and update the trust matrix.
    (Factor_Eve has been removed.)

    Args:
        trust_matrix (np.ndarray): diagonal entries are node trust [0,1.5], off-diagonal entries are
                                   connectivity (0 or 1)
        path_library (list[list[int]]): candidate path library, each path is a list of node indices
        num_multipath (int): number of paths used in this time step
        attack_matrix (np.ndarray): attack-occupation matrix, diagonal 1 means attacked node
        beta_r (float): reward coefficient
        beta_p (float): penalty coefficient
        gama_t (float): temporal recovery coefficient

    Returns:
        np.ndarray: updated trust matrix
    """

    trust_matrix = trust_matrix.copy()
    N = trust_matrix.shape[0]

    # -------------------- Step 1: Generate negative-effect vector --------------------
    # Randomly generate an event for each node: 0=normal, 1=no operation, 2=attack triggered
    rand_vec = np.random.choice([0, 1, 2], size=N)
    # Apply the attack matrix (no Factor_Eve scaling)
    neg_vec = attack_matrix @ rand_vec

    # -------------------- Step 2: Traverse paths and update trust --------------------
    selected_paths = path_library[:min(num_multipath, len(path_library))]

    for path in selected_paths:
        for node in path:
            event = int(neg_vec[node])
            current_trust = trust_matrix[node, node]

            if event == 0:  # reward
                trust_matrix[node, node] += beta_r * (1 - current_trust)
            elif event == 1:  # no operation
                continue
            elif event == 2:  # penalty
                trust_matrix[node, node] -= beta_p * current_trust

            # Clip trust value into [0, 0.999]
            trust_matrix[node, node] = np.clip(trust_matrix[node, node], 0, 0.999)

    # -------------------- Step 3: Temporal recovery effect --------------------
    for i in range(N):
        current_trust = trust_matrix[i, i]
        trust_matrix[i, i] += gama_t * (1 - current_trust)
        trust_matrix[i, i] = np.clip(trust_matrix[i, i], 0, 0.999)

    return trust_matrix


# update attack_matrix
def transfer_attack(attack_matrix, factor_Eve_transfer, topology_matrix):
    """
    Simulate Eve's attack transfer process (at most one attacked node can transfer per step).

    Behavior:
      - Traverse the currently attacked nodes in random order;
      - For the first node that satisfies P > factor_Eve_transfer and has at least one
        unoccupied neighbor, randomly pick a neighbor as the new target:
        set source node to 0 and target node to 1 on the diagonal;
      - If no node satisfies the condition, do nothing;
      - Return the updated attack matrix (the total number of occupied nodes remains unchanged).
    """
    attack_matrix = attack_matrix.copy()
    N = attack_matrix.shape[0]

    attacked_nodes = [i for i in range(N) if attack_matrix[i, i] == 1]
    if not attacked_nodes:
        return attack_matrix  # no attacked nodes, return as is

    # Shuffle to ensure fairness
    random.shuffle(attacked_nodes)

    for node in attacked_nodes:
        P = random.random()
        if P <= factor_Eve_transfer:
            # This node chooses not to transfer (probabilistically stays)
            continue

        # Find neighbors that are not yet attacked
        neighbors = [j for j in range(N)
                     if topology_matrix[node, j] == 1 and attack_matrix[j, j] == 0]
        if not neighbors:
            # Node wants to transfer but has no available neighbor; check next
            continue

        # Randomly select a neighbor as new target and perform the transfer
        new_target = random.choice(neighbors)

        attack_matrix[node, node] = 0
        attack_matrix[new_target, new_target] = 1

        # Only one transfer per step
        return attack_matrix

    # If no transfer happens, return original matrix
    return attack_matrix


# check if attack success
def compute_ASR_single(path, attack_matrix, trust_matrix,
                       occupied_threshold, trust_threshold):
    """
    Determine whether a single transmission is successfully attacked.

    Args:
        path (list[int]): sequence of nodes for the current transmission
        attack_matrix (np.ndarray): attack-occupation matrix, diagonal 1 means node is occupied
        trust_matrix (np.ndarray): trust matrix, diagonal entries are trust values
        occupied_threshold (int): threshold on number of occupied nodes on the path
                                  (>= this → attack success)
        trust_threshold (float): trust threshold (min path trust < this → attack success)

    Returns:
        int: 1 if attack is successful, 0 otherwise
    """

    # Number of nodes on the path that are controlled by Eve
    occupied_count = sum(attack_matrix[node, node] == 1 for node in path)
    # Minimum trust value along the path
    min_trust = min(trust_matrix[node, node] for node in path)

    # Check whether attack conditions are satisfied
    if occupied_count >= occupied_threshold or min_trust < trust_threshold:
        return 1  # attack success
    else:
        return 0  # attack failure


def compute_ASR_iteration(paths,
                          attack_matrix,
                          trust_matrix,
                          occupied_threshold,
                          trust_threshold,
                          asr_mode="key_fraction",
                          loss_per_node=0.3):
    """
    Compute the ASR metric for a single iteration.

    Args:
        paths (list[list[int]]): actual paths used in this iteration,
                                 e.g. selected_paths[:num_multipath]
        attack_matrix (np.ndarray): attack-occupation matrix
        trust_matrix (np.ndarray): trust matrix (used in "binary" mode)
        occupied_threshold (int): threshold used in the old ("binary") ASR definition
        trust_threshold (float): threshold used in the old ("binary") ASR definition
        asr_mode (str):
            - "binary": use the old 0/1 attack success definition
            - "key_fraction": use the new "key segment decay by occupied count" algorithm
        loss_per_node (float):
            - Only used when asr_mode == "key_fraction"
            - For each occupied node on a path, that segment's key loses this fraction,
              remaining ratio is (1 - loss_per_node) ** occupied_count.
              For example, loss_per_node = 0.2 → each occupied node keeps 80%,
              going through 3 occupied nodes keeps 0.8^3.

    Returns:
        float: ASR value for this iteration
            - "binary" mode: 0 or 1 (attack success or failure)
            - "key_fraction" mode: float in [0, 1] (remaining key fraction)
    """
    # Case with no paths
    if not paths:
        if asr_mode == "binary":
            # Original definition: no path → system considered compromised → attack success
            return 1.0
        elif asr_mode == "key_fraction":
            # No path → no secure key → remaining fraction = 0
            return 0.0
        else:
            raise ValueError(f"Unknown asr_mode={asr_mode}")

    # ---- Old binary ASR mode ----
    if asr_mode == "binary":
        flags = []
        for path in paths:
            flag = compute_ASR_single(path, attack_matrix, trust_matrix,
                                      occupied_threshold, trust_threshold)
            flags.append(flag)
        overall_flag = 1 if all(f == 1 for f in flags) else 0
        return float(overall_flag)

    # ---- New "key_fraction" mode: per-path key fraction decays with occupied nodes ----
    elif asr_mode == "key_fraction":
        num_segments = len(paths)
        total_fraction = 0.0

        for path in paths:
            # How many occupied nodes on this path
            occupied_count = sum(attack_matrix[node, node] == 1 for node in path)

            # Each occupied node multiplies the key by (1 - loss_per_node)
            retain_ratio = 1.0 - loss_per_node

            # Sanity clamp (in case of odd configurations)
            if retain_ratio < 0.0:
                retain_ratio = 0.0
            if retain_ratio > 1.0:
                retain_ratio = 1.0

            segment_frac = retain_ratio ** occupied_count
            # If there are no occupied nodes (occupied_count = 0), segment_frac = 1.0

            total_fraction += segment_frac

        # Per-iteration remaining key fraction = average over all segments
        return total_fraction / num_segments if num_segments > 0 else 0.0

    else:
        raise ValueError(f"Unknown asr_mode={asr_mode}")


# simulation
def main_simulation(num_nodes, AP,
                    num_paths, lambda_cor, num_multipath,
                    beta_r, beta_p, gama_t,
                    factor_Eve_transfer,
                    occupied_threshold, trust_threshold,
                    num_iterations, thermalization,
                    attack_matrix_init=None,
                    asr_mode="key_fraction",
                    loss_per_node=0.3
                    ):
    """
    Main simulation function: performs topology generation, path selection,
    trust update, attack spread, and ASR statistics.

    Differences from the older version:
        - No longer takes a fixed topology_matrix as input;
        - Each call randomly generates a new topology_matrix internally
          using generate_connection_matrix.

    Args:
        num_nodes (int): number of nodes
        AP (float): attack pervasiveness (only used when attack_matrix_init is None)
        num_paths (int): size of path library
        lambda_cor (float): path correlation factor
        num_multipath (int): number of simultaneous paths for transmission
        beta_r, beta_p, gama_t (float): trust update parameters
        factor_Eve_transfer (float): attack transfer factor
        occupied_threshold (int): attack success threshold on occupied nodes
        trust_threshold (float): attack success threshold on minimal trust
        num_iterations (int): number of production iterations
        thermalization (int): number of thermalization steps
        attack_matrix_init (np.ndarray, optional): initial attack matrix;
                                                   if None, generate based on AP.

    Returns:
        tuple: (num_multipath, ASR)

        Meaning of ASR:
            - When asr_mode == "binary":
                ASR = attack success rate (mean of 0/1 flags).
            - When asr_mode == "key_fraction":
                First compute the mean "remaining key fraction",
                then return 1 - that value,
                which can still be interpreted as an attack success rate
                (larger ASR → less secure).
    """

    # ---------- Step 0: Randomly generate a topology for each simulation ----------
    topology_matrix = generate_connection_matrix(
        num_nodes=num_nodes,
    )

    # ---------- Step 1: Initialize attack-occupation matrix ----------
    if attack_matrix_init is not None:
        attack_matrix = attack_matrix_init.copy()
    else:
        attack_matrix = generate_attack_matrix(num_nodes, AP)

    # ---------- Step 2: Initialize trust matrix ----------
    trust_matrix = initialize_trust_matrix(topology_matrix)

    # ---------- Initialize statistics array ----------
    arr_AS_Flag = []

    # ---------- Step 3~6: Thermalization phase ----------
    for _ in range(thermalization):
        selected_paths = select_paths(trust_matrix, num_paths, lambda_cor)
        if not selected_paths:
            break  # no valid paths, skip
        trust_matrix = simulate_time_step(trust_matrix, selected_paths, num_multipath,
                                          attack_matrix, beta_r, beta_p, gama_t)
        attack_matrix = transfer_attack(attack_matrix, factor_Eve_transfer, topology_matrix)

    # ---------- Step 7: Production simulation ----------
    for _ in range(num_iterations):
        selected_paths = select_paths(trust_matrix, num_paths, lambda_cor)

        # --------- Case 1: no path at all ---------
        if not selected_paths:
            # Let compute_ASR_iteration handle the difference in modes
            iteration_asr = compute_ASR_iteration(
                paths=[],   # no path
                attack_matrix=attack_matrix,
                trust_matrix=trust_matrix,
                occupied_threshold=occupied_threshold,
                trust_threshold=trust_threshold,
                asr_mode=asr_mode,
                loss_per_node=loss_per_node
            )
            arr_AS_Flag.append(iteration_asr)

            attack_matrix = transfer_attack(attack_matrix, factor_Eve_transfer, topology_matrix)
            continue

        # --------- Normal case: we have paths ---------
        trust_matrix = simulate_time_step(trust_matrix, selected_paths, num_multipath,
                                          attack_matrix, beta_r, beta_p, gama_t)

        # Only use the first num_multipath paths
        used_paths = selected_paths[:num_multipath]

        iteration_asr = compute_ASR_iteration(
            paths=used_paths,
            attack_matrix=attack_matrix,
            trust_matrix=trust_matrix,
            occupied_threshold=occupied_threshold,
            trust_threshold=trust_threshold,
            asr_mode=asr_mode,
            loss_per_node=loss_per_node
        )
        arr_AS_Flag.append(iteration_asr)

        # Attack spread
        attack_matrix = transfer_attack(attack_matrix, factor_Eve_transfer, topology_matrix)

    # ---------- Step 8: Compute ASR ----------
    if arr_AS_Flag:
        mean_val = np.mean(arr_AS_Flag)

        if asr_mode == "key_fraction":
            # In key_fraction mode:
            #   arr_AS_Flag stores "remaining key fraction"
            #   We want to return "attack success rate" = 1 - remaining fraction
            ASR = 1.0 - mean_val
        else:
            # In binary mode:
            #   arr_AS_Flag stores 0/1 attack flags
            #   The mean is directly the attack success rate
            ASR = mean_val
    else:
        ASR = 0.0

    return num_multipath, ASR


def main_simulation_traditional_static_paths_moving_eve(
    num_nodes, AP,
    num_paths, lambda_cor, num_multipath,
    factor_Eve_transfer,
    occupied_threshold, trust_threshold,
    num_iterations, thermalization,
    attack_matrix_init=None,
    asr_mode="key_fraction",
    loss_per_node=0.3
):
    """
    Traditional multipath baseline:
      - Topology is randomly generated;
      - Call select_paths only once to get a path_library;
      - Actual multipath used is fixed: used_paths = path_library[:num_multipath];
      - Does NOT call simulate_time_step → trust_matrix is not updated
        (no Zero-Trust adaptation);
      - Each time step only does:
            compute ASR with fixed paths → Eve transfer (transfer_attack).

    Args are mostly the same as main_simulation, except we remove beta_r, beta_p, gama_t.

    Returns:
        used_paths (list[list[int]]): static multipath set used in this run
        ASR (float): overall attack success rate
        arr_AS_Flag (np.ndarray): array of length num_iterations,
                                  containing the per-step metric ("key_fraction" or "binary")
    """

    # ---------- Step 0: Generate topology ----------
    topology_matrix = generate_connection_matrix(num_nodes=num_nodes)

    # ---------- Step 1: Initialize attack-occupation matrix ----------
    if attack_matrix_init is not None:
        attack_matrix = attack_matrix_init.copy()
    else:
        attack_matrix = generate_attack_matrix(num_nodes, AP)

    # ---------- Step 2: Initialize trust matrix ----------
    # This is only for compatibility with compute_ASR_iteration;
    # in key_fraction mode, the diagonal trust is not used.
    trust_matrix = initialize_trust_matrix(topology_matrix)

    # ---------- Step 3: Select a static path library (distance only) ----------
    # Traditional version: do not use trust_score at all, only hop count
    path_library = select_paths_traditional_distance(
        topology_matrix,
        num_paths=num_paths,
        max_path_len=None
    )

    # If no path can be found, we will pass empty paths to compute_ASR_iteration
    if path_library:
        used_paths = path_library[:min(num_multipath, len(path_library))]
    else:
        used_paths = []

    # ---------- Step 4: Thermalization phase (only move Eve, do not compute ASR) ----------
    for _ in range(thermalization):
        attack_matrix = transfer_attack(attack_matrix, factor_Eve_transfer, topology_matrix)

    # ---------- Step 5: Production simulation ----------
    arr_AS_Flag = []

    for _ in range(num_iterations):
        # Case A: we have static paths
        if used_paths:
            iteration_asr = compute_ASR_iteration(
                paths=used_paths,
                attack_matrix=attack_matrix,
                trust_matrix=trust_matrix,  # not used in key_fraction mode
                occupied_threshold=occupied_threshold,
                trust_threshold=trust_threshold,
                asr_mode=asr_mode,
                loss_per_node=loss_per_node
            )
        # Case B: no path at all
        else:
            iteration_asr = compute_ASR_iteration(
                paths=[],
                attack_matrix=attack_matrix,
                trust_matrix=trust_matrix,
                occupied_threshold=occupied_threshold,
                trust_threshold=trust_threshold,
                asr_mode=asr_mode,
                loss_per_node=loss_per_node
            )

        arr_AS_Flag.append(iteration_asr)

        # After each step, Eve can move on the topology
        attack_matrix = transfer_attack(attack_matrix, factor_Eve_transfer, topology_matrix)

    # ---------- Step 6: Compute overall ASR ----------
    if arr_AS_Flag:
        mean_val = np.mean(arr_AS_Flag)

        if asr_mode == "key_fraction":
            # Here arr_AS_Flag stores "remaining key fraction"
            # We return "attack success rate" = 1 - remaining fraction
            ASR = 1.0 - mean_val
        else:
            # In binary mode arr_AS_Flag stores 0/1 flags
            ASR = mean_val
    else:
        ASR = 0.0

    return used_paths, ASR, np.array(arr_AS_Flag, dtype=float)


def scan_traditional_ASR_vs_AP(num_nodes,
                               num_paths, lambda_cor, num_multipath,
                               factor_Eve_transfer,
                               occupied_threshold, trust_threshold,
                               num_iterations, thermalization,
                               AP_min, AP_max, AP_step,
                               num_repeats=1,
                               asr_mode="key_fraction",
                               loss_per_node=0.3):
    """
    Scan ASR vs AP for the traditional multipath baseline
    (static paths + dynamic Eve).

    Procedure:
      - Fix num_multipath;
      - Sweep AP from AP_min:AP_step:AP_max;
      - For each AP, repeat main_simulation_traditional_static_paths_moving_eve
        num_repeats times and compute mean and std of ASR.

    Returns:
        ap_list (np.ndarray)
        ASR_mean (np.ndarray) shape=(len(ap_list),)
        ASR_std  (np.ndarray) shape=(len(ap_list),)
    """

    if AP_step <= 0:
        raise ValueError("AP_step must be positive")

    # Generate AP list including AP_max (consider floating-point errors)
    ap_list = np.arange(AP_min, AP_max + 1e-12, AP_step)
    ap_list = np.round(ap_list, 12)

    ASR_mean = np.zeros(len(ap_list))
    ASR_std = np.zeros(len(ap_list))

    for j, AP in enumerate(ap_list):
        ASR_list = []

        for r in range(num_repeats):
            # Use different random state each time (no fixed seed here);
            # if you need reproducibility, you can set seed based on (j, r).
            _, ASR_val, _ = main_simulation_traditional_static_paths_moving_eve(
                num_nodes=num_nodes,
                AP=AP,
                num_paths=num_paths,
                lambda_cor=lambda_cor,
                num_multipath=num_multipath,
                factor_Eve_transfer=factor_Eve_transfer,
                occupied_threshold=occupied_threshold,
                trust_threshold=trust_threshold,
                num_iterations=num_iterations,
                thermalization=thermalization,
                attack_matrix_init=None,
                asr_mode=asr_mode,
                loss_per_node=loss_per_node
            )
            ASR_list.append(ASR_val)

        arr = np.array(ASR_list, dtype=float)
        ASR_mean[j] = arr.mean()
        ASR_std[j] = arr.std()

        print(f"[Traditional] AP={AP:.3f}, mean ASR={ASR_mean[j]:.4f}, std={ASR_std[j]:.4f}")

    return ap_list, ASR_mean, ASR_std


# saving and loading results
def save_scan_results(num_nodes, ap_list, mp_range, ASR_mean, ASR_std):
    """
    Save the scan results into a single .npz file (NumPy compressed format).

    Args:
        ap_list (np.ndarray): AP values
        mp_range (np.ndarray): multipath counts
        ASR_mean (np.ndarray): ASR mean matrix
        ASR_std  (np.ndarray): ASR std matrix

    Output:
        Save file into the "raw_data" directory (auto-created if needed).
    """
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"raw_data/n={num_nodes}_{time_str}.npz"
    # Auto-create directory if needed
    directory = os.path.dirname(filename)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)

    np.savez_compressed(
        filename,
        num_nodes=num_nodes,
        ap_list=ap_list,
        mp_range=mp_range,
        ASR_mean=ASR_mean,
        ASR_std=ASR_std
    )

    print(f"[Saved] Data saved to {filename}")


def load_scan_results(filename):
    """
    Load scan results from a saved .npz file.
    """
    data = np.load(filename)
    return (
        data["ap_list"],
        data["mp_range"],
        data["ASR_mean"],
        data["ASR_std"]
    )


# parallel worker
def _worker_scan_point(args):
    """
    Parallel worker: compute ASR mean and std at a given (num_multipath, AP) point.

    Returns:
        (i_mp, j_ap, mean, std)
    """
    (i_mp, j_ap,
     num_nodes,
     AP,
     num_paths, lambda_cor, num_multipath,
     beta_r, beta_p, gama_t,
     factor_Eve_transfer,
     occupied_threshold, trust_threshold,
     num_iterations, thermalization,
     attack_matrix_init,
     num_repeats,
     seed,
     loss_per_node) = args

    # Each process has its own random seed to avoid identical random sequences
    np.random.seed(seed)
    random.seed(seed)

    ASR_list = []
    for _ in range(num_repeats):
        _, ASR = main_simulation(
            num_nodes, AP,
            num_paths, lambda_cor, num_multipath,
            beta_r, beta_p, gama_t,
            factor_Eve_transfer,
            occupied_threshold, trust_threshold,
            num_iterations, thermalization,
            attack_matrix_init=attack_matrix_init,
            asr_mode="key_fraction",
            loss_per_node=loss_per_node
        )
        ASR_list.append(ASR)

    ASR_arr = np.array(ASR_list)
    return i_mp, j_ap, ASR_arr.mean(), ASR_arr.std()


def scan_ASR_table_parallel(num_nodes,
                            num_paths, lambda_cor,
                            beta_r, beta_p, gama_t,
                            factor_Eve_transfer,
                            occupied_threshold, trust_threshold,
                            num_iterations, thermalization,
                            multipath_min, multipath_max, multipath_step,
                            AP_min, AP_max, AP_step,
                            num_repeats=1,
                            num_workers=None,
                            base_seed=None,
                            loss_per_node=0.3):
    """
    Use CPU parallelism to compute the ASR scan table.

    Args:
        num_workers (int or None): number of worker processes.
            - None: ProcessPoolExecutor chooses a default (usually # of CPU cores)

    Returns:
        ap_list (np.ndarray)
        multipath_range (np.ndarray)
        ASR_mean (np.ndarray)  shape = (len(multipath_range), len(ap_list))
        ASR_std  (np.ndarray)  same shape as ASR_mean
    """
    np.random.seed(base_seed)
    random.seed(base_seed)

    # === Generate scan ranges ===
    multipath_range = np.arange(multipath_min, multipath_max + 1e-9,
                                multipath_step, dtype=int)

    # Pre-generate initial attack matrices for all AP values
    ap_list, attack_matrices = generate_attack_matrices_for_AP_range(
        AP_min, AP_max, AP_step, num_nodes
    )
    print(f"[Init] Generated {len(ap_list)} attack matrices for AP range {AP_min}–{AP_max}")

    # Result matrices: rows for num_multipath, columns for AP
    ASR_mean = np.zeros((len(multipath_range), len(ap_list)))
    ASR_std = np.zeros_like(ASR_mean)

    # === Build the parallel task list ===
    tasks = []
    seed_base = int(datetime.datetime.now().timestamp())

    for i_mp, num_multipath in enumerate(multipath_range):
        for j_ap, AP in enumerate(ap_list):
            attack_matrix_init = attack_matrices[j_ap].copy()
            seed = seed_base + i_mp * 1000 + j_ap  # simple seed generation

            args = (
                i_mp, j_ap,
                num_nodes,
                AP,
                num_paths, lambda_cor, num_multipath,
                beta_r, beta_p, gama_t,
                factor_Eve_transfer,
                occupied_threshold, trust_threshold,
                num_iterations, thermalization,
                attack_matrix_init,
                num_repeats,
                seed,
                loss_per_node
            )
            tasks.append(args)

    # === Run in parallel ===
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i_mp, j_ap, mean_val, std_val in executor.map(_worker_scan_point, tasks):
            ASR_mean[i_mp, j_ap] = mean_val
            ASR_std[i_mp, j_ap] = std_val
            print(f"[Summary] mp={multipath_range[i_mp]}, AP={ap_list[j_ap]:.2f}, "
                  f"mean={mean_val:.3f}, std={std_val:.3f}")

    return ap_list, multipath_range, ASR_mean, ASR_std


def main(num_nodes, base_seed):

    start = time.time()

    ap_list, mp_range, ASR_mean, ASR_std = scan_ASR_table_parallel(
        num_nodes=num_nodes,
        num_paths=200,
        lambda_cor=0.2,
        beta_r=0.1,
        beta_p=0.2,
        gama_t=0.005,
        factor_Eve_transfer=0.5,
        occupied_threshold=1,
        trust_threshold=0.7,
        num_iterations=20,
        thermalization=5,
        multipath_min=2,
        multipath_max=6,
        multipath_step=2,
        AP_min=0.45,
        AP_max=0.65,
        AP_step=0.05,
        num_repeats=2,
        num_workers=6,
        base_seed=base_seed,
        loss_per_node=0.2
    )

    end = time.time()
    run_time = end - start

    # Save npz
    save_scan_results(num_nodes, ap_list, mp_range, ASR_mean, ASR_std)


def scan_traditional_multiple_multipaths(
    num_nodes,
    num_paths, lambda_cor,
    multipath_list,                 # e.g. [2, 4, 6]
    factor_Eve_transfer,
    occupied_threshold, trust_threshold,
    num_iterations, thermalization,
    AP_min, AP_max, AP_step,
    num_repeats=1,
    asr_mode="key_fraction",
    loss_per_node=0.3
):
    """
    Scan ASR vs AP for multiple num_multipath values
    for the traditional multipath baseline (static paths + dynamic Eve),
    and store the AP–ASR data.

    Returns:
        results = {
            num_multipath: {
                "ap_list": np.ndarray,
                "ASR_mean": np.ndarray,
                "ASR_std": np.ndarray
            },
            ...
        }
    """

    results = {}

    # Generate AP list
    ap_list = np.arange(AP_min, AP_max + 1e-12, AP_step)
    ap_list = np.round(ap_list, 12)

    for num_multipath in multipath_list:
        print(f"\n===== [Traditional] Scanning num_multipath = {num_multipath} =====")

        ASR_mean = np.zeros(len(ap_list))
        ASR_std = np.zeros(len(ap_list))

        # Sweep AP
        for j, AP in enumerate(ap_list):
            ASR_list = []

            for r in range(num_repeats):
                _, ASR_val, _ = main_simulation_traditional_static_paths_moving_eve(
                    num_nodes=num_nodes,
                    AP=AP,
                    num_paths=num_paths,
                    lambda_cor=lambda_cor,
                    num_multipath=num_multipath,
                    factor_Eve_transfer=factor_Eve_transfer,
                    occupied_threshold=occupied_threshold,
                    trust_threshold=trust_threshold,
                    num_iterations=num_iterations,
                    thermalization=thermalization,
                    attack_matrix_init=None,
                    asr_mode=asr_mode,
                    loss_per_node=loss_per_node
                )
                ASR_list.append(ASR_val)

            arr = np.array(ASR_list, dtype=float)
            ASR_mean[j] = arr.mean()
            ASR_std[j] = arr.std()

            print(f"[Traditional] mp={num_multipath}, AP={AP:.3f}, mean ASR={ASR_mean[j]:.4f}, std={ASR_std[j]:.4f}")

        # Save scan result for this num_multipath
        results[num_multipath] = {
            "ap_list": ap_list.copy(),
            "ASR_mean": ASR_mean.copy(),
            "ASR_std": ASR_std.copy(),
        }

    return results


def save_traditional_multipath_results(num_nodes, results):
    """
    Save the results of scan_traditional_multiple_multipaths into a single npz file.

    results format:
    {
        2: {ap_list, ASR_mean, ASR_std},
        4: {...},
        6: {...}
    }
    """
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"raw_data/traditional_multi_mp_n={num_nodes}_{time_str}.npz"

    directory = os.path.dirname(filename)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)

    # Store all multipath results into the npz
    np.savez_compressed(filename, num_nodes=num_nodes, results=results)

    print(f"\n[Traditional] Saved multipath results to:\n  {filename}")


if __name__ == "__main__":
    num_nodes = 100

    main(num_nodes, None)

    results = scan_traditional_multiple_multipaths(
        num_nodes=num_nodes,
        num_paths=200,
        lambda_cor=0.2,
        multipath_list=[2, 4, 6],     # key point: generate three curves in one run
        factor_Eve_transfer=0.5,
        occupied_threshold=1,
        trust_threshold=0.7,
        num_iterations=20,
        thermalization=5,
        AP_min=0.45,
        AP_max=0.65,
        AP_step=0.05,
        num_repeats=3,
        asr_mode="key_fraction",
        loss_per_node=0.2
    )

    save_traditional_multipath_results(num_nodes, results)
