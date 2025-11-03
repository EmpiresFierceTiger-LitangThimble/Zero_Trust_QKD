import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
# Generate Node Distribition
def random_connection_matrix(num_nodes, connection_prob=0.3):
    """Docstring removed for clarity."""
    upper = np.triu(np.random.rand(num_nodes, num_nodes) < connection_prob, 1)
    adj = upper + upper.T
    return adj.astype(int)
def generate_connection_matrix(num_nodes, connection_prob=0.3):
    while True:
        topology_matrix = random_connection_matrix(num_nodes, connection_prob)
        N = topology_matrix.shape[0]
        head_connected = np.any(topology_matrix[0, 1:])
        tail_connected = np.any(topology_matrix[N - 1, :-1])
        if head_connected and tail_connected:
            break
    return topology_matrix
# Eve choose nodes to attack
def generate_attack_matrix(num_nodes, AP):
    """Docstring removed for clarity."""
    M = np.zeros((num_nodes, num_nodes), dtype=int)
    num_attacked = int(num_nodes * AP)
    possible_nodes = list(range(1, num_nodes - 1))
    num_attacked = min(num_attacked, len(possible_nodes))
    attacked_nodes = random.sample(possible_nodes, num_attacked)
    for node in attacked_nodes:
        M[node, node] = 1
    return M
def generate_attack_matrices_for_AP_range(AP_min, AP_max, AP_step, num_nodes=10):
    """Docstring removed for clarity."""
    if AP_step <= 0:
        raise ValueError("AP_step must be positive")
    ap_list = np.arange(AP_min, AP_max + 1e-12, AP_step)
    ap_list = np.round(ap_list, 12)
    M = len(ap_list)
    N = int(num_nodes)
    if N < 2:
        raise ValueError("num_nodes must be at least 2")
    attack_matrices = np.zeros((M, N, N), dtype=int)
    def _generate_attack_matrix(N, AP):
        mat = np.zeros((N, N), dtype=int)
        k = int(round(AP * N))
        max_nodes = max(0, N - 2)
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
    """Docstring removed for clarity."""
    trust_matrix = topology_matrix.copy().astype(float)
    N = trust_matrix.shape[0]
    for i in range(N):
        trust_matrix[i, i] = 0.999
    return trust_matrix
# Routing (core part of ZTMPQKD)
def select_paths(trust_matrix, num_paths=3, lambda_cor=0.5, max_path_len=None):
    """Docstring removed for clarity."""
    N = trust_matrix.shape[0]
    source, target = 0, N - 1
    G = nx.Graph()
    for i in range(N):
        for j in range(i + 1, N):
            if trust_matrix[i, j] == 1:
                edge_trust = (trust_matrix[i, i] + trust_matrix[j, j]) / 2
                weight = -np.log(edge_trust + 1e-8)
                G.add_edge(i, j, weight=weight, trust=edge_trust)
    all_paths = list(nx.all_simple_paths(G, source=source, target=target))
    total_paths = len(all_paths)
    if total_paths == 0:
        return [], 0
    num_paths = min(num_paths, total_paths)
    all_shortest_paths = list(nx.shortest_simple_paths(G, source, target, weight="weight"))
    candidate_paths = []
    for path in all_shortest_paths:
        if max_path_len and len(path) > max_path_len:
            continue
        candidate_paths.append(path)
        if len(candidate_paths) >= num_paths * 3:
            break
    def path_security(path):
        p = 1.0
        for node in path:
            p *= trust_matrix[node, node]
        return p
    sec_values = [path_security(p) for p in candidate_paths]
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
    avg_cor = np.mean(cor_values) if cor_values else 0
    TH_Cor = lambda_cor * avg_cor
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
    """Docstring removed for clarity."""
    trust_matrix = trust_matrix.copy()
    N = trust_matrix.shape[0]
    rand_vec = np.random.choice([0, 1, 2], size=N)
    neg_vec = attack_matrix @ rand_vec
    selected_paths = path_library[:min(num_multipath, len(path_library))]
    for path in selected_paths:
        for node in path:
            event = int(neg_vec[node])
            current_trust = trust_matrix[node, node]
            if event == 0:
                trust_matrix[node, node] += beta_r * (1 - current_trust)
            elif event == 1:
                continue
            elif event == 2:
                trust_matrix[node, node] -= beta_p * current_trust
            trust_matrix[node, node] = np.clip(trust_matrix[node, node], 0, 0.999)
    for i in range(N):
        current_trust = trust_matrix[i, i]
        trust_matrix[i, i] += gama_t * (1 - current_trust)
        trust_matrix[i, i] = np.clip(trust_matrix[i, i], 0, 0.999)
    return trust_matrix
# update attack_matrix
def transfer_attack(attack_matrix, factor_Eve_transfer, topology_matrix):
    """Docstring removed for clarity."""
    attack_matrix = attack_matrix.copy()
    N = attack_matrix.shape[0]
    attacked_nodes = [i for i in range(N) if attack_matrix[i, i] == 1]
    if not attacked_nodes:
        return attack_matrix
    random.shuffle(attacked_nodes)
    for node in attacked_nodes:
        P = random.random()
        if P <= factor_Eve_transfer:
            continue
        neighbors = [j for j in range(N)
                     if topology_matrix[node, j] == 1 and attack_matrix[j, j] == 0]
        if not neighbors:
            continue
        new_target = random.choice(neighbors)
        attack_matrix[node, node] = 0
        attack_matrix[new_target, new_target] = 1
        return attack_matrix
    return attack_matrix
# check if attack success
def compute_ASR_single(path, attack_matrix, trust_matrix, 
                       occupied_threshold, trust_threshold):
    """Docstring removed for clarity."""
    occupied_count = sum(attack_matrix[node, node] == 1 for node in path)
    min_trust = min(trust_matrix[node, node] for node in path)
    if occupied_count >= occupied_threshold or min_trust < trust_threshold:
        return 1
    else:
        return 0
# simulation
def main_simulation(topology_matrix, num_nodes, AP,
                    num_paths, lambda_cor, num_multipath,
                    beta_r, beta_p, gama_t,
                    factor_Eve_transfer,
                    occupied_threshold, trust_threshold,
                    num_iterations, thermalization,
                    attack_matrix_init=None):
    """Docstring removed for clarity."""
    if attack_matrix_init is not None:
        attack_matrix = attack_matrix_init.copy()
    else:
        attack_matrix = generate_attack_matrix(num_nodes, AP)
    trust_matrix = initialize_trust_matrix(topology_matrix)
    arr_AS_Flag = []
    for _ in range(thermalization):
        selected_paths = select_paths(trust_matrix, num_paths, lambda_cor)
        if not selected_paths:
            break
        trust_matrix = simulate_time_step(trust_matrix, selected_paths, num_multipath,
                                          attack_matrix, beta_r, beta_p, gama_t)
        attack_matrix = transfer_attack(attack_matrix, factor_Eve_transfer, topology_matrix)
    for _ in range(num_iterations):
        selected_paths = select_paths(trust_matrix, num_paths, lambda_cor)
        if not selected_paths:
            arr_AS_Flag.append(1)
            attack_matrix = transfer_attack(attack_matrix, factor_Eve_transfer, topology_matrix)
            continue
        trust_matrix = simulate_time_step(trust_matrix, selected_paths, num_multipath,
                                          attack_matrix, beta_r, beta_p, gama_t)
        flags = []
        for path in selected_paths[:num_multipath]:
            flag = compute_ASR_single(path, attack_matrix, trust_matrix,
                                      occupied_threshold, trust_threshold)
            flags.append(flag)
        overall_flag = 1 if all(f == 1 for f in flags) else 0
        arr_AS_Flag.append(overall_flag)
        attack_matrix = transfer_attack(attack_matrix, factor_Eve_transfer, topology_matrix)
    ASR = np.mean(arr_AS_Flag) if arr_AS_Flag else 0.0
    return num_multipath, ASR
def scan_and_plot(topology_matrix, num_nodes,
                  num_paths, lambda_cor,
                  beta_r, beta_p, gama_t,
                  factor_Eve_transfer,
                  occupied_threshold, trust_threshold,
                  num_iterations, thermalization,
                  multipath_min, multipath_max, multipath_step,
                  AP_min, AP_max, AP_step):
    """Docstring removed for clarity."""
    multipath_range = np.arange(multipath_min, multipath_max + 1e-9, multipath_step, dtype=int)
    ap_list, attack_matrices = generate_attack_matrices_for_AP_range(AP_min, AP_max, AP_step, num_nodes)
    print(f"[Init] Generated {len(ap_list)} attack matrices for AP range {AP_min}–{AP_max}")
    plt.figure(figsize=(8, 6))
    color_map = plt.cm.get_cmap("viridis", len(multipath_range))
    for idx, num_multipath in enumerate(multipath_range):
        ASR_curve = []
        for ap_idx, AP in enumerate(ap_list):
            attack_matrix_init = attack_matrices[ap_idx].copy()
            _, ASR = main_simulation(
                topology_matrix=topology_matrix,
                num_nodes=num_nodes,
                AP=AP,
                num_paths=num_paths,
                lambda_cor=lambda_cor,
                num_multipath=num_multipath,
                beta_r=beta_r,
                beta_p=beta_p,
                gama_t=gama_t,
                factor_Eve_transfer=factor_Eve_transfer,
                occupied_threshold=occupied_threshold,
                trust_threshold=trust_threshold,
                num_iterations=num_iterations,
                thermalization=thermalization,
                attack_matrix_init=attack_matrix_init
            )
            ASR_curve.append(ASR)
            print(f"[Scan] num_multipath={num_multipath}, AP={AP:.2f}, ASR={ASR:.3f}")
        plt.plot(ap_list, ASR_curve, "o-",
                 color=color_map(idx),
                 label=f"num_multipath={num_multipath}")
    plt.xlabel("Attacker Pervasiveness (AP)")
    plt.ylabel("Average Attack Success Rate (ASR)")
    plt.title("ASR vs AP for Different Multipath Counts (Fixed Initial Attack Matrices)")
    plt.grid(True)
    plt.legend(title="Multipath")
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    num_nodes = 10
    topology_matrix = generate_connection_matrix(num_nodes=num_nodes, connection_prob=0.4)
    scan_and_plot(
    topology_matrix,
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
    AP_min=0.0,
    AP_max=0.8,
    AP_step=0.2
)
