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
    生成无向随机连接矩阵。
    参数:
        num_nodes (int): 节点数量
        connection_prob (float): 两节点之间建立连接的概率 (0~1)
    返回:
        np.ndarray: num_nodes × num_nodes 的方阵，A[i][j]=1 表示节点 i 和 j 相连，否则为 0
    """
    # 随机生成上三角部分
    upper = np.triu(np.random.rand(num_nodes, num_nodes) < connection_prob, 1)
    # 对称化得到无向图矩阵
    adj = upper + upper.T
    # 转换为 int 类型（0 或 1）
    return adj.astype(int)

def generate_connection_matrix(num_nodes, connection_prob=0.3):
    while True:
        topology_matrix = random_connection_matrix(num_nodes, connection_prob)
        N = topology_matrix.shape[0]

        # 检查起点和终点是否与至少一个节点相连
        head_connected = np.any(topology_matrix[0, 1:])  # 第0行除自己以外有1
        tail_connected = np.any(topology_matrix[N - 1, :-1])

        if head_connected and tail_connected:
            break  # 满足条件则退出循环，否则重建拓扑
    return topology_matrix

# Eve choose nodes to attack
def generate_attack_matrix(num_nodes, AP):
    """
    生成攻击占据矩阵。
    参数:
        num_nodes (int): 节点数量
        AP (float): 攻击渗透率 (0~1)，表示被攻击节点的比例
    返回:
        np.ndarray: num_nodes × num_nodes 方阵，A[i][i]=1 表示节点 i 被占据，否则为 0
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
    生成一组攻击占据矩阵（对角线表示被占据节点为1），对应从 AP_min 到 AP_max 的每个 AP 值。
    起点(0)与终点(num_nodes-1)永远不会被选为被占据节点。

    参数:
        AP_min (float): AP 的起始值（包含）
        AP_max (float): AP 的结束值（包含）
        AP_step (float): AP 的步进值
        num_nodes (int): 节点总数，默认 10

    返回:
        tuple:
            ap_list (np.ndarray): 所有 AP 值
            attack_matrices (np.ndarray): 形如 (len(ap_list), num_nodes, num_nodes) 的整型矩阵数组
    """

    if AP_step <= 0:
        raise ValueError("AP_step must be positive")

    # 生成包含 AP_max 的 AP 列表
    ap_list = np.arange(AP_min, AP_max + 1e-12, AP_step)
    ap_list = np.round(ap_list, 12)

    M = len(ap_list)
    N = int(num_nodes)
    if N < 2:
        raise ValueError("num_nodes must be at least 2")

    attack_matrices = np.zeros((M, N, N), dtype=int)

    # 局部函数：根据 AP 生成攻击矩阵
    def _generate_attack_matrix(N, AP):
        mat = np.zeros((N, N), dtype=int)
        k = int(round(AP * N))
        max_nodes = max(0, N - 2)  # 排除首尾节点
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
    根据拓扑关系矩阵生成初始信任度矩阵。
    对角线信任度全部为 1，其余元素保持拓扑连接关系。

    参数:
        topology_matrix (np.ndarray): 拓扑关系矩阵，A[i,j]=1 表示节点i与节点j相连

    返回:
        np.ndarray: 初始信任度矩阵，对角线为1，非对角为拓扑关系(0或1)
    """
    trust_matrix = topology_matrix.copy().astype(float)
    N = trust_matrix.shape[0]

    # 对角线信任度全部设为1
    for i in range(N):
        trust_matrix[i, i] = 0.999

    return trust_matrix


# Routing (core part of ZTMPQKD)

'''
# old
def select_paths(trust_matrix, num_paths=3, lambda_cor=0.5, max_path_len=None):
    """
    根据信任度矩阵选择最优路径。
    第0个节点固定为传输起点，最后一个节点固定为终点。
    
    参数:
        trust_matrix (np.ndarray): 方阵，对角线为节点信任度[0,1]，其余元素0或1表示连接关系
        num_paths (int): 要选择的路径数量(K)
        lambda_cor (float): 相关性阈值比例 λCor (0~1)
        max_path_len (int or None): 限制最大路径长度
    
    返回:
        list[list[int]]: 最优路径列表（如果无路可达，则返回 []）
    """

    N = trust_matrix.shape[0]
    source, target = 0, N - 1  # 固定起点和终点

    G = nx.Graph()

    # --- 构建图 ---
    for i in range(N):
        for j in range(i + 1, N):
            if trust_matrix[i, j] == 1:
                # 边的信任度 = 两端节点信任度平均值
                edge_trust = (trust_matrix[i, i] + trust_matrix[j, j]) / 2
                # 转换为加权距离（越可信越短）
                weight = -np.log(edge_trust + 1e-8)
                G.add_edge(i, j, weight=weight, trust=edge_trust)

    # --- 计算所有可行路径 ---
    all_paths = list(nx.all_simple_paths(G, source=source, target=target))
    total_paths = len(all_paths)

    if total_paths == 0:
        # 统一：无路可达时只返回一个空列表
        return []

    # 限制 num_paths 不超过最大可行路径数
    num_paths = min(num_paths, total_paths)

    # --- 寻找候选路径 (按信任度最优优先) ---
    all_shortest_paths = list(nx.shortest_simple_paths(G, source, target, weight="weight"))
    candidate_paths = []
    for path in all_shortest_paths:
        if max_path_len and len(path) > max_path_len:
            continue
        candidate_paths.append(path)
        if len(candidate_paths) >= num_paths * 3:
            break

    # --- 路径安全性 ---
    def path_security(path):
        p = 1.0
        for node in path:
            p *= trust_matrix[node, node]
        return p

    sec_values = [path_security(p) for p in candidate_paths]

    # --- 路径相关性 ---
    def path_correlation(p1, p2):
        edges1 = set(tuple(sorted((p1[i], p1[i + 1]))) for i in range(len(p1) - 1))
        edges2 = set(tuple(sorted((p2[i], p2[i + 1]))) for i in range(len(p2) - 1))
        common = len(edges1 & edges2)
        total = len(edges1 | edges2)
        return common / total if total else 0.0

    # --- 计算平均相关性与阈值 ---
    cor_values = []
    for i in range(len(candidate_paths)):
        for j in range(i + 1, len(candidate_paths)):
            cor_values.append(path_correlation(candidate_paths[i], candidate_paths[j]))
    avg_cor = np.mean(cor_values) if cor_values else 0
    TH_Cor = lambda_cor * avg_cor

    # --- 贪心筛选路径 ---
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
'''

def select_paths(trust_matrix,
                 num_paths=3,
                 lambda_cor=0.5,
                 max_path_len=None,
                 max_combination_paths=20):
    """
    根据信任度矩阵选择最优路径。
    第0个节点固定为传输起点，最后一个节点固定为终点。

    参数:
        trust_matrix (np.ndarray): 方阵，对角线为节点信任度[0,1]，
                                   其余元素0或1表示连接关系
        num_paths (int): 要选择的路径数量(K)
        lambda_cor (float): 相关性阈值比例 λCor (0~1)
        max_path_len (int or None): 限制最大路径长度（以“节点数”为单位）
        max_combination_paths (int or None): 进入“相关性+组合筛选”阶段的
                                             最大候选路径数上限。
                                             None 表示不限制。

    返回:
        list[list[int]]: 最优路径列表（如果无路可达，则返回 []）
    """

    N = trust_matrix.shape[0]
    source, target = 0, N - 1

    G = nx.Graph()

    # --- 构建图（边权重由两端节点信任度决定） ---
    for i in range(N):
        for j in range(i + 1, N):
            if trust_matrix[i, j] == 1:
                # 边的信任度 = 两端节点信任度平均值
                edge_trust = (trust_matrix[i, i] + trust_matrix[j, j]) / 2
                # 转换为加权距离（越可信越短）
                weight = -np.log(edge_trust + 1e-8)
                G.add_edge(i, j, weight=weight, trust=edge_trust)

    # --- 计算所有可行路径数量（只用来限制 num_paths） ---
    all_paths = list(nx.all_simple_paths(G, source=source, target=target))
    total_paths = len(all_paths)

    if total_paths == 0:
        # 无路可达
        return []

    # 限制 num_paths 不超过最大可行路径数
    num_paths = min(num_paths, total_paths)

    # --- 寻找候选路径 (按最短“加权距离”优先) ---
    all_shortest_paths = list(nx.shortest_simple_paths(
        G, source, target, weight="weight"
    ))
    candidate_paths = []
    for path in all_shortest_paths:
        if max_path_len and len(path) > max_path_len:
            continue
        candidate_paths.append(path)
        # 保持和你原来一样：多取一些候选（这里是 num_paths * 3）
        if len(candidate_paths) >= num_paths * 3:
            break

    if not candidate_paths:
        return []

    # --- 只在“组合筛选阶段”限制候选路径数量 ---
    if (max_combination_paths is not None
            and len(candidate_paths) > max_combination_paths):
        candidate_paths = candidate_paths[:max_combination_paths]

    # --- 路径安全性 ---
    def path_security(path):
        p = 1.0
        for node in path:
            p *= trust_matrix[node, node]
        return p

    sec_values = [path_security(p) for p in candidate_paths]

    # --- 路径相关性 ---
    def path_correlation(p1, p2):
        edges1 = set(
            tuple(sorted((p1[i], p1[i + 1]))) for i in range(len(p1) - 1)
        )
        edges2 = set(
            tuple(sorted((p2[i], p2[i + 1]))) for i in range(len(p2) - 1)
        )
        common = len(edges1 & edges2)
        total = len(edges1 | edges2)
        return common / total if total else 0.0

    # --- 计算平均相关性与阈值 ---
    cor_values = []
    for i in range(len(candidate_paths)):
        for j in range(i + 1, len(candidate_paths)):
            cor_values.append(path_correlation(
                candidate_paths[i], candidate_paths[j]
            ))
    avg_cor = np.mean(cor_values) if cor_values else 0.0
    TH_Cor = lambda_cor * avg_cor

    # --- 贪心筛选路径 ---
    selected = []
    for path, sec in sorted(
        zip(candidate_paths, sec_values),
        key=lambda x: -x[1]
    ):
        if not selected:
            selected.append(path)
        else:
            corr_ok = all(
                path_correlation(path, p_sel) < TH_Cor
                for p_sel in selected
            )
            if corr_ok:
                selected.append(path)
        if len(selected) >= num_paths:
            break

    return selected


# single step of transmission (update trust matrix)

def simulate_time_step(trust_matrix, path_library, num_multipath,
                       attack_matrix, beta_r, beta_p, gama_t):
    """
    模拟一个时间步的多路径量子传输过程，并更新信任度矩阵。
    （移除了 Factor_Eve 参数）

    参数:
        trust_matrix (np.ndarray): 对角线为节点信任度[0,1.5]，其余为连通关系(0或1)
        path_library (list[list[int]]): 备选路径库，每条路径为节点索引列表
        num_multipath (int): 当前时间步选取的多路径数量
        attack_matrix (np.ndarray): 攻击占据矩阵，对角线1表示被攻击节点
        beta_r (float): 奖励系数
        beta_p (float): 惩罚系数
        gama_t (float): 时间恢复系数

    返回:
        np.ndarray: 更新后的信任度矩阵
    """

    trust_matrix = trust_matrix.copy()
    N = trust_matrix.shape[0]

    # -------------------- Step 1: 生成负面效应向量 --------------------
    # 随机生成每个节点事件：0=正常，1=无操作，2=攻击触发
    rand_vec = np.random.choice([0, 1, 2], size=N)
    # 攻击矩阵作用结果（去掉 Factor_Eve 缩放）
    neg_vec = attack_matrix @ rand_vec

    # -------------------- Step 2: 路径遍历与信任更新 --------------------
    selected_paths = path_library[:min(num_multipath, len(path_library))]

    for path in selected_paths:
        for node in path:
            event = int(neg_vec[node])
            current_trust = trust_matrix[node, node]

            if event == 0:  # 奖励
                trust_matrix[node, node] += beta_r * (1 - current_trust)
            elif event == 1:  # 无操作
                continue
            elif event == 2:  # 惩罚
                trust_matrix[node, node] -= beta_p * current_trust

            # 信任值范围裁剪 → [0, 1.5]
            trust_matrix[node, node] = np.clip(trust_matrix[node, node], 0, 0.999)

    # -------------------- Step 3: 时间恢复效应 --------------------
    for i in range(N):
        current_trust = trust_matrix[i, i]
        trust_matrix[i, i] += gama_t * (1 - current_trust)
        trust_matrix[i, i] = np.clip(trust_matrix[i, i], 0, 0.999)

    return trust_matrix

# update attack_matrix

def transfer_attack(attack_matrix, factor_Eve_transfer, topology_matrix):
    """
    模拟 Eve 的攻击转移过程（每次最多只允许一个被攻占节点转移）。

    行为：
      - 在当前被攻占节点集合中按随机顺序逐个检查；
      - 对于第一个满足 P > factor_Eve_transfer 且存在未被占据邻居的节点，
        随机选一个邻居作为转移目标：将源节点设为 0，目标节点设为 1；
      - 若没有任何节点满足条件，则不做修改；
      - 返回更新后的攻击矩阵（不会改变被占据节点总数）。
    """
    attack_matrix = attack_matrix.copy()
    N = attack_matrix.shape[0]

    attacked_nodes = [i for i in range(N) if attack_matrix[i, i] == 1]
    if not attacked_nodes:
        return attack_matrix  # 无被攻占节点，直接返回

    # 按随机顺序遍历，以保证公平性
    random.shuffle(attacked_nodes)

    for node in attacked_nodes:
        P = random.random()
        if P <= factor_Eve_transfer:
            # 该节点不进行转移（概率上保留）
            continue

        # 找到未被占据的邻居（排除已被占据的）
        neighbors = [j for j in range(N)
                     if topology_matrix[node, j] == 1 and attack_matrix[j, j] == 0]
        if not neighbors:
            # 虽然该节点想转移，但没有可用邻居，继续检查下一个被攻占节点
            continue

        # 随机选择一个邻居作为转移目标，并执行转移（一次性仅允许一个转移）
        new_target = random.choice(neighbors)

        attack_matrix[node, node] = 0
        attack_matrix[new_target, new_target] = 1

        # 一次只做一个转移，直接返回
        return attack_matrix

    # 若循环结束都没有转移发生，返回原矩阵（未改变）
    return attack_matrix


# check if attack success
def compute_ASR_single(path, attack_matrix, trust_matrix, 
                       occupied_threshold, trust_threshold):
    """
    判断单次传输是否被攻击成功。
    
    参数:
        path (list[int]): 当前传输所经过的节点序列
        attack_matrix (np.ndarray): 攻击占据矩阵，对角线1表示节点被Eve攻占
        trust_matrix (np.ndarray): 信任度矩阵，对角线为节点信任度
        occupied_threshold (int): 攻占数量阈值（>=该数量则攻击成功）
        trust_threshold (float): 信任度阈值（低于该值则攻击成功）
    
    返回:
        int: 攻击成功为1，攻击失败为0
    """
    
    # 路径中被Eve控制的节点数量
    occupied_count = sum(attack_matrix[node, node] == 1 for node in path)
    # 路径中最低信任度
    min_trust = min(trust_matrix[node, node] for node in path)

    # 判断是否满足攻击成功条件
    if occupied_count >= occupied_threshold or min_trust < trust_threshold:
        return 1  # 攻击成功
    else:
        return 0  # 攻击失败


# simulation
'''
def main_simulation(topology_matrix, num_nodes, AP,
                    num_paths, lambda_cor, num_multipath,
                    beta_r, beta_p, gama_t,
                    factor_Eve_transfer,
                    occupied_threshold, trust_threshold,
                    num_iterations, thermalization,
                    attack_matrix_init=None):
    """
    主模拟函数：执行路径选择、信任更新、攻击传播与 ASR 统计。
    已修改以支持从外部传入固定初始攻击占据矩阵。

    参数:
        topology_matrix (np.ndarray): 固定拓扑关系矩阵
        num_nodes (int): 节点数量
        AP (float): 攻击渗透率（仅在未提供 attack_matrix_init 时使用）
        num_paths (int): 备选路线库数量
        lambda_cor (float): 路径相关度系数
        num_multipath (int): 多路径传输数量
        beta_r, beta_p, gama_t (float): 信任更新参数
        factor_Eve_transfer (float): 攻击转移系数
        occupied_threshold (int): 攻占数量阈值
        trust_threshold (float): 信任度阈值
        num_iterations (int): 正式模拟步数
        thermalization (int): 热化步数
        attack_matrix_init (np.ndarray, optional): 外部传入的初始攻击矩阵
                                                   若为 None，则自动生成

    返回:
        tuple: (num_multipath, ASR)
    """

    # ---------- Step 1: 初始化攻击占据矩阵 ----------
    if attack_matrix_init is not None:
        attack_matrix = attack_matrix_init.copy()
    else:
        attack_matrix = generate_attack_matrix(num_nodes, AP)

    # ---------- Step 2: 初始化信任度矩阵 ----------
    trust_matrix = initialize_trust_matrix(topology_matrix)

    # ---------- 初始化统计数组 ----------
    arr_AS_Flag = []

    # ---------- Step 3~6: 热化阶段 ----------
    for _ in range(thermalization):
        selected_paths = select_paths(trust_matrix, num_paths, lambda_cor)
        if not selected_paths:
            break  # 无有效路径时跳过
        trust_matrix = simulate_time_step(trust_matrix, selected_paths, num_multipath,
                                          attack_matrix, beta_r, beta_p, gama_t)
        attack_matrix = transfer_attack(attack_matrix, factor_Eve_transfer, topology_matrix)

    # ---------- Step 7: 正式模拟 ----------
    for _ in range(num_iterations):
        selected_paths = select_paths(trust_matrix, num_paths, lambda_cor)
        if not selected_paths:
            # 若路径断裂，视为一次失败（系统被攻破）
            arr_AS_Flag.append(1)
            attack_matrix = transfer_attack(attack_matrix, factor_Eve_transfer, topology_matrix)
            continue

        trust_matrix = simulate_time_step(trust_matrix, selected_paths, num_multipath,
                                          attack_matrix, beta_r, beta_p, gama_t)

        # 计算系统级攻击结果（所有路径均失败记为成功攻击）
        flags = []
        for path in selected_paths[:num_multipath]:
            flag = compute_ASR_single(path, attack_matrix, trust_matrix,
                                      occupied_threshold, trust_threshold)
            flags.append(flag)
        overall_flag = 1 if all(f == 1 for f in flags) else 0
        arr_AS_Flag.append(overall_flag)

        # 攻击扩散
        attack_matrix = transfer_attack(attack_matrix, factor_Eve_transfer, topology_matrix)

    # ---------- Step 8: 计算 ASR ----------
    ASR = np.mean(arr_AS_Flag) if arr_AS_Flag else 0.0

    return num_multipath, ASR
'''

def main_simulation(num_nodes, AP,
                    num_paths, lambda_cor, num_multipath,
                    beta_r, beta_p, gama_t,
                    factor_Eve_transfer,
                    occupied_threshold, trust_threshold,
                    num_iterations, thermalization,
                    connection_prob=0.3,
                    attack_matrix_init=None):
    """
    主模拟函数：执行拓扑生成、路径选择、信任更新、攻击传播与 ASR 统计。
    与旧版的区别：
        - 不再从外部传入固定的 topology_matrix；
        - 每次调用时，内部用 generate_connection_matrix 随机生成一个拓扑矩阵。

    参数:
        num_nodes (int): 节点数量
        AP (float): 攻击渗透率（仅在未提供 attack_matrix_init 时使用）
        num_paths (int): 备选路线库数量
        lambda_cor (float): 路径相关度系数
        num_multipath (int): 多路径传输数量
        beta_r, beta_p, gama_t (float): 信任更新参数
        factor_Eve_transfer (float): 攻击转移系数
        occupied_threshold (int): 攻占数量阈值
        trust_threshold (float): 信任度阈值
        num_iterations (int): 正式模拟步数
        thermalization (int): 热化步数
        connection_prob (float): 生成随机拓扑时的连边概率
        attack_matrix_init (np.ndarray, optional): 外部传入的初始攻击矩阵；
                                                   若为 None，则根据 AP 自动生成

    返回:
        tuple: (num_multipath, ASR)
    """

    # ---------- Step 0: 每次模拟都随机生成一个拓扑矩阵 ----------
    topology_matrix = generate_connection_matrix(
        num_nodes=num_nodes,
        connection_prob=connection_prob
    )

    # ---------- Step 1: 初始化攻击占据矩阵 ----------
    if attack_matrix_init is not None:
        attack_matrix = attack_matrix_init.copy()
    else:
        attack_matrix = generate_attack_matrix(num_nodes, AP)

    # ---------- Step 2: 初始化信任度矩阵 ----------
    trust_matrix = initialize_trust_matrix(topology_matrix)

    # ---------- 初始化统计数组 ----------
    arr_AS_Flag = []

    # ---------- Step 3~6: 热化阶段 ----------
    for _ in range(thermalization):
        selected_paths = select_paths(trust_matrix, num_paths, lambda_cor)
        if not selected_paths:
            break  # 无有效路径时跳过
        trust_matrix = simulate_time_step(trust_matrix, selected_paths, num_multipath,
                                          attack_matrix, beta_r, beta_p, gama_t)
        attack_matrix = transfer_attack(attack_matrix, factor_Eve_transfer, topology_matrix)

    # ---------- Step 7: 正式模拟 ----------
    for _ in range(num_iterations):
        selected_paths = select_paths(trust_matrix, num_paths, lambda_cor)
        if not selected_paths:
            # 若路径断裂，视为一次失败（系统被攻破）
            arr_AS_Flag.append(1)
            attack_matrix = transfer_attack(attack_matrix, factor_Eve_transfer, topology_matrix)
            continue

        trust_matrix = simulate_time_step(trust_matrix, selected_paths, num_multipath,
                                          attack_matrix, beta_r, beta_p, gama_t)

        # 计算系统级攻击结果（所有路径均失败记为成功攻击）
        flags = []
        for path in selected_paths[:num_multipath]:
            flag = compute_ASR_single(path, attack_matrix, trust_matrix,
                                      occupied_threshold, trust_threshold)
            flags.append(flag)
        overall_flag = 1 if all(f == 1 for f in flags) else 0
        arr_AS_Flag.append(overall_flag)

        # 攻击扩散
        attack_matrix = transfer_attack(attack_matrix, factor_Eve_transfer, topology_matrix)

    # ---------- Step 8: 计算 ASR ----------
    ASR = np.mean(arr_AS_Flag) if arr_AS_Flag else 0.0

    return num_multipath, ASR

def scan_ASR_table(num_nodes,
                   num_paths, lambda_cor,
                   beta_r, beta_p, gama_t,
                   factor_Eve_transfer,
                   occupied_threshold, trust_threshold,
                   num_iterations, thermalization,
                   multipath_min, multipath_max, multipath_step,
                   AP_min, AP_max, AP_step,
                   num_repeats=1,
                   connection_prob=0.3):
    """
    在不同的 AP 和不同的 num_multipath 下调用 main_simulation，
    计算 ASR，并以表格形式返回 (均值 + 标准差)。

    参数：
        num_nodes (int): 节点数量
        num_paths (int): 路径库大小 (select_paths 的候选路径数 K)
        lambda_cor (float): 路径相关度系数
        beta_r, beta_p, gama_t (float): 信任更新参数
        factor_Eve_transfer (float): 攻击转移系数
        occupied_threshold (int): 攻占节点个数阈值
        trust_threshold (float): 信任度阈值
        num_iterations (int): 每次模拟的时间步数
        thermalization (int): 热化步数
        multipath_min, multipath_max, multipath_step (int): num_multipath 扫描范围
        AP_min, AP_max, AP_step (float): AP 扫描范围
        num_repeats (int): 同一 (AP, num_multipath) 下重复模拟次数
        connection_prob (float): 生成随机拓扑时的连边概率

    返回：
        ap_list (np.ndarray): 所有 AP 取值
        multipath_range (np.ndarray): 所有 num_multipath 取值
        ASR_mean (np.ndarray): 形状 (len(multipath_range), len(ap_list)) 的均值矩阵
        ASR_std  (np.ndarray): 同形状的标准差矩阵
    """

    # === 生成扫描范围 ===
    multipath_range = np.arange(multipath_min, multipath_max + 1e-9,
                                multipath_step, dtype=int)

    # 为所有 AP 预生成初始攻击矩阵（保证相同 AP 下初始攻击分布一致）
    ap_list, attack_matrices = generate_attack_matrices_for_AP_range(
        AP_min, AP_max, AP_step, num_nodes
    )
    print(f"[Init] Generated {len(ap_list)} attack matrices for AP range {AP_min}–{AP_max}")

    # 结果矩阵：行对应 num_multipath，列对应 AP
    ASR_mean = np.zeros((len(multipath_range), len(ap_list)))
    ASR_std  = np.zeros_like(ASR_mean)

    # === 外层：遍历 num_multipath ===
    for i_mp, num_multipath in enumerate(multipath_range):

        # === 内层：遍历 AP ===
        for j_ap, AP in enumerate(ap_list):
            attack_matrix_init = attack_matrices[j_ap].copy()

            ASR_list = []

            # === 最内层：重复模拟 num_repeats 次 ===
            for rep in range(num_repeats):
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
                print(f"[Scan] mp={num_multipath}, AP={AP:.2f}, "
                      f"rep={rep+1}/{num_repeats}, ASR={ASR:.3f}")

            ASR_arr = np.array(ASR_list)
            ASR_mean[i_mp, j_ap] = ASR_arr.mean()
            ASR_std[i_mp, j_ap]  = ASR_arr.std()

            print(f"[Summary] mp={num_multipath}, AP={AP:.2f}, "
                  f"mean={ASR_mean[i_mp, j_ap]:.3f}, std={ASR_std[i_mp, j_ap]:.3f}")

    return ap_list, multipath_range, ASR_mean, ASR_std


# 保存和加载模拟结果
def save_scan_results(num_nodes, ap_list, mp_range, ASR_mean, ASR_std):
    """
    将扫描得到的数据保存到单个 .npz 文件中（numpy 压缩格式）。

    参数:
        ap_list (np.ndarray): AP 取值数组
        mp_range (np.ndarray): 多路径数量数组
        ASR_mean (np.ndarray): ASR 均值矩阵
        ASR_std  (np.ndarray): ASR 标准差矩阵
        filename (str): 输出文件名（应当以 .npz 结尾）

    输出:
        保存文件到当前目录下
    """
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"raw_data/n={num_nodes}_{time}.npz"
    # 自动创建目录（若用户希望放到子目录，如 "results/xxx.npz"）
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
    从保存的 .npz 文件加载扫描结果
    """
    data = np.load(filename)
    return (
        data["ap_list"],
        data["mp_range"],
        data["ASR_mean"],
        data["ASR_std"]
    )

# 生成 Attack Success rate vs. Attack Pervasiveness
def plot_AS_vs_AP_from_npz(filename,
                           show_std=True,
                           save_fig=True,
                           out_dir="plots"):
    """
    从 .npz 文件绘制 ASR vs AP 曲线。
    
    参数:
        filename (str): .npz 文件路径
        show_std (bool): 是否绘制误差棒
        save_fig (bool): 是否保存图像
        out_dir (str): 保存目录
    """

    # --- 1. 读取数据 ---
    data = np.load(filename)
    ap_list = data["ap_list"]
    mp_range = data["mp_range"]
    ASR_mean = data["ASR_mean"]
    ASR_std = data["ASR_std"]

    # --- 2. 绘图 ---
    plt.figure(figsize=(8,5))
    
    for i, mp in enumerate(mp_range):
        if show_std:
            plt.errorbar(
                ap_list, ASR_mean[i],
                yerr=ASR_std[i],
                label=f"num_multipath = {mp}",
                fmt="-o", capsize=4
            )
        else:
            plt.plot(
                ap_list, ASR_mean[i],
                "-o", label=f"num_multipath = {mp}"
            )

    plt.xlabel("Attack Pervasiveness (AP)")
    plt.ylabel("ASR")
    plt.title("ASR vs AP")
    plt.grid(True)
    plt.legend()

    # --- 3. 保存 ---
    if save_fig:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = os.path.join(out_dir, f"AS_vs_AP_{time}.png")
        plt.savefig(out_path, dpi=300)
        print(f"[Saved] Figure saved to {out_path}")

    plt.show()

# 并行
def _worker_scan_point(args):
    """
    并行 worker：计算某一个 (num_multipath, AP) 点上的 ASR 均值和标准差。
    返回 (i_mp, j_ap, mean, std)
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

    # 每个进程内独立随机种子，避免不同 worker 完全相同的随机序列
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
    使用 CPU 并行来计算 ASR 扫描表格。

    参数基本与 scan_ASR_table 相同，新增：
        num_workers (int or None): 并行进程数。
            - None: 默认由 ProcessPoolExecutor 决定（通常是 CPU 核数）

    返回：
        ap_list (np.ndarray)
        multipath_range (np.ndarray)
        ASR_mean (np.ndarray)  shape = (len(multipath_range), len(ap_list))
        ASR_std  (np.ndarray)  同上
    """

    # === 生成扫描范围 ===
    multipath_range = np.arange(multipath_min, multipath_max + 1e-9,
                                multipath_step, dtype=int)

    # 为所有 AP 预生成初始攻击矩阵（保证相同 AP 下初始攻击分布一致）
    ap_list, attack_matrices = generate_attack_matrices_for_AP_range(
        AP_min, AP_max, AP_step, num_nodes
    )
    print(f"[Init] Generated {len(ap_list)} attack matrices for AP range {AP_min}–{AP_max}")

    # 结果矩阵：行对应 num_multipath，列对应 AP
    ASR_mean = np.zeros((len(multipath_range), len(ap_list)))
    ASR_std  = np.zeros_like(ASR_mean)

    # === 准备并行任务列表 ===
    tasks = []
    seed_base = int(datetime.datetime.now().timestamp())

    for i_mp, num_multipath in enumerate(multipath_range):
        for j_ap, AP in enumerate(ap_list):
            attack_matrix_init = attack_matrices[j_ap].copy()
            seed = seed_base + i_mp * 1000 + j_ap  # 简单生成不同 seed

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

    # === 并行执行 ===
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i_mp, j_ap, mean_val, std_val in executor.map(_worker_scan_point, tasks):
            ASR_mean[i_mp, j_ap] = mean_val
            ASR_std[i_mp, j_ap]  = std_val
            print(f"[Summary] mp={multipath_range[i_mp]}, AP={ap_list[j_ap]:.2f}, "
                  f"mean={mean_val:.3f}, std={std_val:.3f}")

    return ap_list, multipath_range, ASR_mean, ASR_std


if __name__ == "__main__":
    num_nodes = 12

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
        AP_max=0.8,
        AP_step=0.1,
        num_repeats=3,
        connection_prob=0.3,
        num_workers=None
    )
    

    end = time.time()
    print(f"{end-start}")
    save_scan_results(num_nodes, ap_list, mp_range, ASR_mean, ASR_std)
