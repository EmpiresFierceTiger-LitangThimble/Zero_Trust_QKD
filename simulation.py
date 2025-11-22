import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import datetime
import os
import concurrent.futures
import time


# Generate Node Distribition
def random_connection_matrix(num_nodes, connection_prob=0.3):
    """
    Generate an undirected random connection matrix.

    Parameters:
        num_nodes (int): Number of nodes
        connection_prob (float): Probability of establishing a connection between two nodes (0–1)

    Returns:
        np.ndarray: A num_nodes × num_nodes matrix where A[i][j] = 1 means node i and j are connected, otherwise 0
    """
    # Randomly generate the upper triangular part
    upper = np.triu(np.random.rand(num_nodes, num_nodes) < connection_prob, 1)
    # Symmetrize to obtain an undirected adjacency matrix
    adj = upper + upper.T
    # Convert to int type (0 or 1)
    return adj.astype(int)

def generate_connection_matrix(num_nodes, connection_prob=0.3):
    while True:
        topology_matrix = random_connection_matrix(num_nodes, connection_prob)
        N = topology_matrix.shape[0]

        # Check whether the start node and end node are connected to at least one other node
        head_connected = np.any(topology_matrix[0, 1:])  # Row 0 has a 1 outside of itself
        tail_connected = np.any(topology_matrix[N - 1, :-1])

        # Exit loop if the condition is satisfied; otherwise regenerate the topology
        if head_connected and tail_connected:
            break
    return topology_matrix

# Eve chooses nodes to attack
def generate_attack_matrix(num_nodes, AP):
    """
    Generate an attack occupancy matrix.

    Parameters:
        num_nodes (int): Number of nodes
        AP (float): Attack pervasiveness (0–1), representing the proportion of nodes that are attacked

    Returns:
        np.ndarray: A num_nodes × num_nodes matrix where A[i][i] = 1 means node i is occupied (attacked), otherwise 0
    """
    M = np.zeros((num_nodes, num_nodes), dtype=int)
    num_attacked = int(num_nodes * AP)

    # Internal nodes that Eve can attack (excluding first and last)
    possible_nodes = list(range(1, num_nodes - 1))

    # Ensure attacked count does not exceed available nodes
    num_attacked = min(num_attacked, len(possible_nodes))

    # Randomly choose nodes to attack
    attacked_nodes = random.sample(possible_nodes, num_attacked)

    # Fill diagonal to indicate which nodes are attacked
    for node in attacked_nodes:
        M[node, node] = 1

    return M

def generate_attack_matrices_for_AP_range(AP_min, AP_max, AP_step, num_nodes=10):
    """
    Generate a set of attack occupancy matrices (diagonal entries = 1 indicate attacked nodes),
    corresponding to each AP value from AP_min to AP_max.
    The start node (0) and end node (num_nodes - 1) will never be chosen as attacked nodes.

    Parameters:
        AP_min (float): Starting AP value (inclusive)
        AP_max (float): Ending AP value (inclusive)
        AP_step (float): Increment step for AP
        num_nodes (int): Total number of nodes, default is 10

    Returns:
        tuple:
            ap_list (np.ndarray): All AP values
            attack_matrices (np.ndarray): An integer matrix array of shape
                                          (len(ap_list), num_nodes, num_nodes)
    """

    if AP_step <= 0:
        raise ValueError("AP_step must be positive")

    # Generate AP list including AP_max
    ap_list = np.arange(AP_min, AP_max + 1e-12, AP_step)
    ap_list = np.round(ap_list, 12)

    M = len(ap_list)
    N = int(num_nodes)
    if N < 2:
        raise ValueError("num_nodes must be at least 2")

    attack_matrices = np.zeros((M, N, N), dtype=int)

    # Local function: generate an attack matrix based on AP
    def _generate_attack_matrix(N, AP):
        mat = np.zeros((N, N), dtype=int)
        k = int(round(AP * N))
        max_nodes = max(0, N - 2)  # Exclude the first and last nodes
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

# Initialize trust_matrix
def initialize_trust_matrix(topology_matrix):
    """
    Generate an initial trust matrix based on the topology matrix.
    All diagonal trust values are set to 1, and all other entries follow the topology
    (1 for connected nodes, 0 for not connected).

    Parameters:
        topology_matrix (np.ndarray): Topology matrix where A[i, j] = 1 means
                                      node i is connected to node j.

    Returns:
        np.ndarray: Initial trust matrix with diagonal entries set to 1 and
                    off-diagonal entries equal to the topology values (0 or 1).
    """
    trust_matrix = topology_matrix.copy().astype(float)
    N = trust_matrix.shape[0]

    # Set all diagonal trust values to 1
    for i in range(N):
        trust_matrix[i, i] = 0.999

    return trust_matrix


# Routing (core part of ZTMPQKD)


# old
def select_paths(trust_matrix, num_paths=3, lambda_cor=0.5, max_path_len=None):
    """
    Select optimal paths based on the trust matrix.
    Node 0 is fixed as the source, and the last node is fixed as the target.
    
    Parameters:
        trust_matrix (np.ndarray): Square matrix where diagonal entries are node trust values in [0,1],
                                   and off-diagonal entries are 0/1 indicating connectivity.
        num_paths (int): Number of paths to select (K).
        lambda_cor (float): Correlation threshold ratio λCor (0–1).
        max_path_len (int or None): Maximum allowed path length.
    
    Returns:
        list[list[int]]: List of optimal paths (returns [] if no path is reachable).
    """

    N = trust_matrix.shape[0]
    source, target = 0, N - 1  # Fixed source and target

    G = nx.Graph()

    # --- Build graph ---
    for i in range(N):
        for j in range(i + 1, N):
            if trust_matrix[i, j] == 1:
                # Edge trust = average trust of the two endpoint nodes
                edge_trust = (trust_matrix[i, i] + trust_matrix[j, j]) / 2
                # Convert to weighted distance (more trustworthy means shorter)
                weight = -np.log(edge_trust + 1e-8)
                G.add_edge(i, j, weight=weight, trust=edge_trust)

    # --- Compute all feasible paths ---
    all_paths = list(nx.all_simple_paths(G, source=source, target=target))
    total_paths = len(all_paths)

    if total_paths == 0:
        # Unified behavior: return only an empty list when no path is reachable
        return []

    # Limit num_paths to not exceed the number of feasible paths
    num_paths = min(num_paths, total_paths)

    # --- Find candidate paths (prioritize highest trust / shortest weighted paths) ---
    all_shortest_paths = list(nx.shortest_simple_paths(G, source, target, weight="weight"))
    candidate_paths = []
    for path in all_shortest_paths:
        if max_path_len and len(path) > max_path_len:
            continue
        candidate_paths.append(path)
        if len(candidate_paths) >= num_paths * 3:
            break

    # --- Path security ---
    def path_security(path):
        p = 1.0
        for node in path:
            p *= trust_matrix[node, node]
        return p

    sec_values = [path_security(p) for p in candidate_paths]

    # --- Path correlation ---
    def path_correlation(p1, p2):
        edges1 = set(tuple(sorted((p1[i], p1[i + 1]))) for i in range(len(p1) - 1))
        edges2 = set(tuple(sorted((p2[i], p2[i + 1]))) for i in range(len(p2) - 1))
        common = len(edges1 & edges2)
        total = len(edges1 | edges2)
        return common / total if total else 0.0

    # --- Compute average correlation and threshold ---
    cor_values = []
    for i in range(len(candidate_paths)):
        for j in range(i + 1, len(candidate_paths)):
            cor_values.append(path_correlation(candidate_paths[i], candidate_paths[j]))
    avg_cor = np.mean(cor_values) if cor_values else 0
    TH_Cor = lambda_cor * avg_cor

    # --- Greedy path selection ---
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
    Simulate one time step of the multipath quantum transmission process and update
    the trust matrix.
    (The Factor_Eve parameter has been removed.)

    Parameters:
        trust_matrix (np.ndarray): Diagonal entries are node trust values in [0, 1.5],
                                   off-diagonal entries are connectivity (0 or 1).
        path_library (list[list[int]]): Candidate path library, each path is a list of node indices.
        num_multipath (int): Number of multipaths selected at the current time step.
        attack_matrix (np.ndarray): Attack occupancy matrix; diagonal = 1 indicates an attacked node.
        beta_r (float): Reward coefficient.
        beta_p (float): Penalty coefficient.
        gama_t (float): Time-recovery coefficient.

    Returns:
        np.ndarray: Updated trust matrix.
    """

    trust_matrix = trust_matrix.copy()
    N = trust_matrix.shape[0]

    # -------------------- Step 1: Generate negative-effect vector --------------------
    # Randomly generate an event for each node: 0 = normal, 1 = no operation, 2 = attack triggered
    rand_vec = np.random.choice([0, 1, 2], size=N)
    # Apply attack matrix (without Factor_Eve scaling)
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

            # Clip trust value to range [0, 1.5]
            trust_matrix[node, node] = np.clip(trust_matrix[node, node], 0, 0.999)

    # -------------------- Step 3: Time-recovery effect --------------------
    for i in range(N):
        current_trust = trust_matrix[i, i]
        trust_matrix[i, i] += gama_t * (1 - current_trust)
        trust_matrix[i, i] = np.clip(trust_matrix[i, i], 0, 0.999)

    return trust_matrix

# update attack_matrix

def transfer_attack(attack_matrix, factor_Eve_transfer, topology_matrix):
    """
    Simulate Eve's attack transfer process (at most one occupied node is allowed to transfer per step).

    Behavior:
      - Randomly shuffle the currently occupied nodes and check them one by one;
      - For the first node that satisfies P > factor_Eve_transfer and has at least one unoccupied neighbor,
        randomly choose one neighbor as the transfer target: set the source node to 0 and the target node to 1;
      - If no node satisfies the condition, do nothing;
      - Return the updated attack matrix (the total number of occupied nodes is unchanged).
    """
    attack_matrix = attack_matrix.copy()
    N = attack_matrix.shape[0]

    attacked_nodes = [i for i in range(N) if attack_matrix[i, i] == 1]
    if not attacked_nodes:
        return attack_matrix  # No occupied nodes, return directly

    # Traverse in random order to ensure fairness
    random.shuffle(attacked_nodes)

    for node in attacked_nodes:
        P = random.random()
        if P <= factor_Eve_transfer:
            # This node does not transfer (kept with this probability)
            continue

        # Find unoccupied neighbors (exclude already occupied ones)
        neighbors = [
            j for j in range(N)
            if topology_matrix[node, j] == 1 and attack_matrix[j, j] == 0
        ]
        if not neighbors:
            # The node wants to transfer but has no available neighbor; check the next occupied node
            continue

        # Randomly choose a neighbor as the transfer target and perform the transfer
        # (only one transfer is allowed per step)
        new_target = random.choice(neighbors)

        attack_matrix[node, node] = 0
        attack_matrix[new_target, new_target] = 1

        # Only one transfer per step; return immediately
        return attack_matrix

    # If no transfer happens after checking all nodes, return the original matrix (unchanged)
    return attack_matrix


# check if attack success
def compute_ASR_single(path, attack_matrix, trust_matrix, 
                       occupied_threshold, trust_threshold):
    """
    Determine whether a single transmission is successfully attacked.

    Parameters:
        path (list[int]): Sequence of nodes along the current transmission path.
        attack_matrix (np.ndarray): Attack occupancy matrix; diagonal = 1 indicates the node is occupied by Eve.
        trust_matrix (np.ndarray): Trust matrix; diagonal entries are node trust values.
        occupied_threshold (int): Occupied-node threshold (attack succeeds if occupied_count >= this value).
        trust_threshold (float): Trust threshold (attack succeeds if min_trust < this value).

    Returns:
        int: 1 if the attack succeeds, 0 if the attack fails.
    """

    # Number of nodes on the path that are controlled by Eve
    occupied_count = sum(attack_matrix[node, node] == 1 for node in path)
    # Minimum trust value along the path
    min_trust = min(trust_matrix[node, node] for node in path)

    # Check whether the attack success conditions are met
    if occupied_count >= occupied_threshold or min_trust < trust_threshold:
        return 1  # Attack succeeds
    else:
        return 0  # Attack fails


# simulation
def main_simulation(num_nodes, AP,
                    num_paths, lambda_cor, num_multipath,
                    beta_r, beta_p, gama_t,
                    factor_Eve_transfer,
                    occupied_threshold, trust_threshold,
                    num_iterations, thermalization,
                    connection_prob=0.3,
                    attack_matrix_init=None):
    """
    Main simulation function: performs topology generation, path selection, trust update,
    attack propagation, and ASR statistics.
    Differences from the old version:
        - No longer takes a fixed topology_matrix from outside;
        - Each call randomly generates a topology matrix internally using
          generate_connection_matrix.

    Parameters:
        num_nodes (int): Number of nodes.
        AP (float): Attack pervasiveness (used only when attack_matrix_init is not provided).
        num_paths (int): Number of candidate paths in the path library.
        lambda_cor (float): Path correlation coefficient.
        num_multipath (int): Number of paths used in multipath transmission.
        beta_r, beta_p, gama_t (float): Trust update parameters.
        factor_Eve_transfer (float): Attack transfer coefficient.
        occupied_threshold (int): Occupied-node threshold.
        trust_threshold (float): Trust threshold.
        num_iterations (int): Number of main simulation steps.
        thermalization (int): Number of thermalization steps.
        connection_prob (float): Edge probability when generating a random topology.
        attack_matrix_init (np.ndarray, optional): Externally provided initial attack matrix;
                                                   if None, it will be generated from AP.

    Returns:
        tuple: (num_multipath, ASR)
    """

    # ---------- Step 0: Randomly generate a topology matrix for each simulation run ----------
    topology_matrix = generate_connection_matrix(
        num_nodes=num_nodes,
        connection_prob=connection_prob
    )

    # ---------- Step 1: Initialize the attack occupancy matrix ----------
    if attack_matrix_init is not None:
        attack_matrix = attack_matrix_init.copy()
    else:
        attack_matrix = generate_attack_matrix(num_nodes, AP)

    # ---------- Step 2: Initialize the trust matrix ----------
    trust_matrix = initialize_trust_matrix(topology_matrix)

    # ---------- Initialize statistics array ----------
    arr_AS_Flag = []

    # ---------- Step 3~6: Thermalization phase ----------
    for _ in range(thermalization):
        selected_paths = select_paths(trust_matrix, num_paths, lambda_cor)
        if not selected_paths:
            break  # Skip if no valid path exists
        trust_matrix = simulate_time_step(trust_matrix, selected_paths, num_multipath,
                                          attack_matrix, beta_r, beta_p, gama_t)
        attack_matrix = transfer_attack(attack_matrix, factor_Eve_transfer, topology_matrix)

    # ---------- Step 7: Main simulation ----------
    for _ in range(num_iterations):
        selected_paths = select_paths(trust_matrix, num_paths, lambda_cor)
        if not selected_paths:
            # If paths are broken, count as a failure (system compromised)
            arr_AS_Flag.append(1)
            attack_matrix = transfer_attack(attack_matrix, factor_Eve_transfer, topology_matrix)
            continue

        trust_matrix = simulate_time_step(trust_matrix, selected_paths, num_multipath,
                                          attack_matrix, beta_r, beta_p, gama_t)

        # Compute system-level attack outcome (attack is successful only if all paths fail)
        flags = []
        for path in selected_paths[:num_multipath]:
            flag = compute_ASR_single(path, attack_matrix, trust_matrix,
                                      occupied_threshold, trust_threshold)
            flags.append(flag)
        overall_flag = 1 if all(f == 1 for f in flags) else 0
        arr_AS_Flag.append(overall_flag)

        # Attack diffusion
        attack_matrix = transfer_attack(attack_matrix, factor_Eve_transfer, topology_matrix)

    # ---------- Step 8: Compute ASR ----------
    ASR = np.mean(arr_AS_Flag) if arr_AS_Flag else 0.0

    return num_multipath, ASR

# Save and load simulation results
def save_scan_results(num_nodes, ap_list, mp_range, ASR_mean, ASR_std):
    """
    Save the scanned simulation data into a single .npz file (NumPy compressed format).

    Parameters:
        ap_list (np.ndarray): Array of AP values.
        mp_range (np.ndarray): Array of multipath counts.
        ASR_mean (np.ndarray): Matrix of ASR mean values.
        ASR_std  (np.ndarray): Matrix of ASR standard deviation values.

    Output:
        Saves the file in the current directory (or a subdirectory if specified).
    """
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"raw_data/n={num_nodes}_{time}.npz"

    # Automatically create directory if needed (e.g., "results/xxx.npz")
    directory = os.path.dirname(filename)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)

    np.savez_compressed(
        filename,
        ap_list=ap_list,
        mp_range=mp_range,
        ASR_mean=ASR_mean,
        ASR_std=ASR_std
    )

    print(f"[Saved] Data saved to {filename}")


def load_scan_results(filename):
    """
    Load scanned simulation results from a saved .npz file.
    """
    data = np.load(filename)
    return (
        data["ap_list"],
        data["mp_range"],
        data["ASR_mean"],
        data["ASR_std"]
    )


# Parallel
def _worker_scan_point(args):
    """
    Parallel worker: compute the ASR mean and standard deviation for one
    (num_multipath, AP) scan point.
    Returns (i_mp, j_ap, mean, std).
    """
    (i_mp, j_ap,
     num_nodes,
     AP,
     num_paths, lambda_cor, num_multipath,
     beta_r, beta_p, gama_t,
     factor_Eve_transfer,
     occupied_threshold, trust_threshold,
     num_iterations, thermalization,
     connection_prob,
     attack_matrix_init,
     num_repeats,
     seed) = args

    # Independent random seed per process to avoid identical random sequences across workers
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
            connection_prob=connection_prob,
            attack_matrix_init=attack_matrix_init
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
                            connection_prob=0.3,
                            num_workers=None):
    """
    Compute the ASR scan table using CPU parallelism.

    Parameters are basically the same as scan_ASR_table, with one additional argument:
        num_workers (int or None): Number of parallel processes.
            - None: Let ProcessPoolExecutor decide (usually the CPU core count).

    Returns:
        ap_list (np.ndarray)
        multipath_range (np.ndarray)
        ASR_mean (np.ndarray): shape = (len(multipath_range), len(ap_list))
        ASR_std  (np.ndarray): same shape as ASR_mean
    """

    # === Generate scan ranges ===
    multipath_range = np.arange(multipath_min, multipath_max + 1e-9,
                                multipath_step, dtype=int)

    # Pre-generate initial attack matrices for all AP values
    # (ensures the same initial attack distribution for the same AP)
    ap_list, attack_matrices = generate_attack_matrices_for_AP_range(
        AP_min, AP_max, AP_step, num_nodes
    )
    print(f"[Init] Generated {len(ap_list)} attack matrices for AP range {AP_min}–{AP_max}")

    # Result matrices: rows correspond to num_multipath, columns correspond to AP
    ASR_mean = np.zeros((len(multipath_range), len(ap_list)))
    ASR_std  = np.zeros_like(ASR_mean)

    # === Prepare parallel task list ===
    tasks = []
    seed_base = int(datetime.datetime.now().timestamp())

    for i_mp, num_multipath in enumerate(multipath_range):
        for j_ap, AP in enumerate(ap_list):
            attack_matrix_init = attack_matrices[j_ap].copy()
            seed = seed_base + i_mp * 1000 + j_ap  # Simple way to generate different seeds

            args = (
                i_mp, j_ap,
                num_nodes,
                AP,
                num_paths, lambda_cor, num_multipath,
                beta_r, beta_p, gama_t,
                factor_Eve_transfer,
                occupied_threshold, trust_threshold,
                num_iterations, thermalization,
                connection_prob,
                attack_matrix_init,
                num_repeats,
                seed
            )
            tasks.append(args)

    # === Run in parallel ===
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i_mp, j_ap, mean_val, std_val in executor.map(_worker_scan_point, tasks):
            ASR_mean[i_mp, j_ap] = mean_val
            ASR_std[i_mp, j_ap]  = std_val
            print(f"[Summary] mp={multipath_range[i_mp]}, AP={ap_list[j_ap]:.2f}, "
                  f"mean={mean_val:.3f}, std={std_val:.3f}")

    return ap_list, multipath_range, ASR_mean, ASR_std

def log_run(log_path, num_nodes, params, run_time,
            ap_list, mp_range, ASR_mean, ASR_std):
    """
    Write a log entry for a single run.
    The log is opened in append mode, so old records will not be overwritten.

    Parameters:
        log_path (str): Path to the log file.
        num_nodes (int): Number of nodes.
        params (dict): All parameters used in the simulation (as a dictionary).
        run_time (float): Runtime of this run (seconds).
        ap_list, mp_range, ASR_mean, ASR_std: Scan results.
    """
    # Automatically create the directory if it doesn't exist
    log_dir = os.path.dirname(log_path)
    if log_dir != "" and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Current timestamp
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"[Run Timestamp]: {ts}\n")
        f.write(f"[Num Nodes]: {num_nodes}\n")
        f.write("[Parameters]:\n")
        for k, v in params.items():
            f.write(f"    {k} = {v}\n")

        f.write("\n[Results]:\n")
        f.write(f"    ap_list = {ap_list.tolist()}\n")
        f.write(f"    mp_range = {mp_range.tolist()}\n")

        # Write only the mean values (optionally write more)
        f.write(f"    ASR_mean = {ASR_mean.tolist()}\n")
        f.write(f"    ASR_std  = {ASR_std.tolist()}\n")

        f.write(f"\n[Run Time]: {run_time:.2f} seconds\n")
        f.write("="*80 + "\n")


def main():
    log_path = "logs/simulation_log.txt"
    for num_nodes in range(13, 16):
        for i in range(3):
            start = time.time()

            ap_list, mp_range, ASR_mean, ASR_std = scan_ASR_table_parallel(
                num_nodes=num_nodes,
                num_paths=8,
                lambda_cor=0.5,
                beta_r=0.1,
                beta_p=0.3,
                gama_t=0.005,
                factor_Eve_transfer=0.03,
                occupied_threshold=1,
                trust_threshold=0.7,
                num_iterations=10**3,
                thermalization=100,
                multipath_min=2,
                multipath_max=6,
                multipath_step=2,
                AP_min=0.1,
                AP_max=0.6,
                AP_step=0.1,
                num_repeats=3,
                connection_prob=0.3,
                num_workers=None
            )

            end = time.time()
            run_time = end - start

            # Save npz
            save_scan_results(num_nodes, ap_list, mp_range, ASR_mean, ASR_std)

            # --- New: log this run ---
            params = {
                "num_paths": 8,
                "lambda_cor": 0.5,
                "beta_r": 0.1,
                "beta_p": 0.3,
                "gama_t": 0.005,
                "factor_Eve_transfer": 0.03,
                "occupied_threshold": 1,
                "trust_threshold": 0.7,
                "num_iterations": 10**3,
                "thermalization": 100,
                "multipath_min": 2,
                "multipath_max": 6,
                "multipath_step": 2,
                "AP_min": 0.1,
                "AP_max": 0.6,
                "AP_step": 0.1,
                "num_repeats": 3,
                "connection_prob": 0.3,
                "num_workers": None
            }

            log_run(log_path, num_nodes, params, run_time,
                    ap_list, mp_range, ASR_mean, ASR_std)


if __name__ == "__main__":
    main()
