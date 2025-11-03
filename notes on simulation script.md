# **Notes on Simulation Code** (Summarized by ChatGPT)

## **Overview**

The simulation code implements a Python-based framework for evaluating the security and resilience of a **Zero-Trust Multipath Quantum Key Distribution (ZT-MP-QKD)** network.
It models how node trust, attack propagation, and path diversification interact dynamically to produce quantitative estimates of the **Attack Success Rate (ASR)** under varying network and threat conditions.

The framework combines probabilistic routing, adaptive trust management, and dynamic adversarial movement to emulate **spatiotemporal diversification** in a quantum communication network.

---

## **1. File Structure and Execution**

* **File name:** `main_1.py`
* **Main entry:** `if __name__ == "__main__":`
* **Primary function:** `scan_and_plot()` — performs two-level scanning over **attack penetration rate (AP)** and **number of multipath transmissions**.
* **Output:**

  * Console logs showing progress (e.g., `[Scan] num_multipath=4, AP=0.40, ASR=0.753`)
  * Automatically generated **ASR vs. AP** plots for comparison.

---

## **2. Core Modules**

### **2.1 Network Initialization**

**Functions:**
`random_connection_matrix()` and `generate_connection_matrix()`

* Generate a random undirected network topology with `num_nodes`.
* Each pair of nodes is connected with probability `connection_prob`.
* Ensure both the source node (0) and destination node (N–1) have at least one connection.
* Output: a binary **adjacency matrix** representing connectivity.

---

### **2.2 Attack Matrix Generation**

**Functions:**
`generate_attack_matrix()` and `generate_attack_matrices_for_AP_range()`

* Create diagonal **attack matrices** marking compromised nodes (`1` = attacked, `0` = secure).
* The **attack penetration rate (AP)** controls the initial fraction of occupied nodes.
* Source and destination nodes are always excluded from attack initialization.
* For parameter sweeps, multiple attack matrices are pre-generated across the AP range (0.0–0.8).

---

### **2.3 Trust Matrix Initialization**

**Function:**
`initialize_trust_matrix()`

* Builds the **trust matrix** `T`, initialized from the topology.
* Diagonal entries `T[i,i]` represent per-node trust levels (≈1.0 at start).
* Off-diagonal entries inherit network connections from the adjacency matrix.
* The trust matrix evolves over time based on system events.

---

### **2.4 Path Selection Algorithm**

**Function:**
`select_paths()`

* Constructs a weighted graph using current trust values.

* Each edge weight is computed as:

  ```math
  w_{ij} = -\ln\!\left(\frac{T_{ii}+T_{jj}}{2} + \varepsilon\right)
  ```

* Enumerates feasible routes between source and destination using a **modified Dijkstra algorithm**.

* Evaluates path security as the product of trust values along the path:

  ```math
  P(S_i) = \prod_{j \in S_i} T_{jj}
  ```

* Calculates **path correlation** between routes:

  ```math
  \text{Cor}(S_i,S_j) = \frac{|E_i \cap E_j|}{|E_i \cup E_j|}
  ```

* A correlation threshold (`λ_cor`) ensures low overlap and promotes route diversity.

* Returns a set of optimal multipath routes for secure transmission.

---

### **2.5 Time-Step Simulation**

**Function:**
`simulate_time_step()`

* Simulates one time step of multipath transmission.
* Each node experiences random events:

  * **Reward:** trust increases by `β_r * (1 - T)`.
  * **Penalty:** trust decreases by `β_p * T`.
  * **Recovery:** trust increases gradually via `γ_t`.
* Attack influence is modeled through the **attack matrix**, which amplifies negative effects on compromised nodes.
* Trust values are bounded within `[0, 1]` after updates.

---

### **2.6 Attack Propagation**

**Function:**
`transfer_attack()`

* Models **Eve’s attack transfer** between connected nodes.
* For each compromised node:

  * With probability `factor_Eve_transfer`, the attack may spread to one neighboring node.
* Only one transfer is allowed per iteration to maintain localized propagation.
* The total number of attacked nodes remains constant.

---

### **2.7 Attack Evaluation**

**Function:**
`compute_ASR_single()`

* Determines if a transmission path is compromised based on two conditions:

  1. Number of attacked nodes ≥ `occupied_threshold`.
  2. Minimum trust value on path < `trust_threshold`.
* Returns binary outcome (1 = compromised, 0 = secure).
* Aggregates over all paths and time steps to compute the **Attack Success Rate (ASR)**.

---

### **2.8 Main Simulation Routine**

**Function:**
`main_simulation()`

* Combines all subsystems into a single run:

  1. Initialize trust and attack states.
  2. Run a **thermalization phase** to stabilize trust evolution.
  3. Execute iterative transmissions and update trust and attack matrices.
  4. Record per-step outcomes to compute ASR.
* Returns `(num_multipath, ASR)` for each parameter configuration.

---

### **2.9 Parameter Scanning and Visualization**

**Function:**
`scan_and_plot()`

* Performs a two-dimensional parameter sweep:

  * Outer loop: number of multipaths (2–6).
  * Inner loop: attack penetration rate (0.0–0.8).
* Uses pre-generated attack matrices to ensure fairness across AP values.
* Plots **ASR vs. AP** for all configurations using Matplotlib’s *viridis* colormap.

---

## **3. Key Parameters**

| Parameter                      | Description                                | Typical Value     |
| ------------------------------ | ------------------------------------------ | ----------------- |
| `num_nodes`                    | Number of nodes in the network             | 10                |
| `connection_prob`              | Probability of edge creation               | 0.4               |
| `num_paths`                    | Number of candidate routes                 | 8                 |
| `lambda_cor`                   | Path correlation factor                    | 0.5               |
| `beta_r` / `beta_p` / `gama_t` | Reward, penalty, and recovery coefficients | 0.1 / 0.3 / 0.005 |
| `factor_Eve_transfer`          | Probability of attack spreading            | 0.03              |
| `occupied_threshold`           | Attack success node threshold              | 1                 |
| `trust_threshold`              | Minimum trust threshold                    | 0.7               |
| `num_iterations`               | Total iterations                           | 1000              |
| `thermalization`               | Warm-up iterations                         | 100               |
| `AP_min–AP_max`                | Attack pervasiveness range                 | 0.0 – 0.8         |
| `multipath_min–multipath_max`  | Multipath range                            | 2 – 6             |

---

## **4. Simulation Workflow Summary**

1. **Generate Topology** – Create a connected random network.
2. **Initialize States** – Define initial attack and trust matrices.
3. **Select Paths** – Use trust-weighted routing with correlation control.
4. **Iterate Simulation** – Update trust and propagate attacks per time step.
5. **Compute ASR** – Evaluate the proportion of compromised transmissions.
6. **Visualize** – Plot ASR trends against AP and multipath count.

---

## **5. Interpretation**

The simulation results quantify the balance between multipath redundancy and adversarial influence.
At **low AP values**, the benefit of multiple transmission paths is minimal since most nodes remain uncompromised.
As **AP increases**, additional paths significantly improve network resilience by spreading key exchanges across independent routes.
However, at **high AP levels (≥0.8)**, nearly all nodes are compromised, causing ASR to approach unity and nullifying the advantage of route diversification.
