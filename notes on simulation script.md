Default to not trusting any node (whether it's a relay or within the boundary), and assume threats exist everywhere;

Real-time dynamic routing and Moving Target Defense (MTD) (continuously changing communication paths and parameters to increase the difficulty of attacks);

Probabilistic and penalty-based trust management mechanisms (continuously adjusting each node's trust level instead of using static assignments);

Centralized control with global monitoring (similar to the control plane in a Zero Trust architecture).


module 1 (Transmission)
    submodule 1 (Path Selection)
        1. Use Yen's algorithm to find a set of shortest paths S(C_p)
        2. Calculate correlation Cor for every pair of paths, P_i and P_j, in S(C_p). Compare Cor with a threshold value Cor_max,
        if Cor < Cor_max, proceed.
            a. Cor_max = lambda_cor * Cor_avg_in_C_p
        3. Calculate delta_secure of adding P_i and P_j, if delta_secure > 0, add P_i and P_j to S(P).
            a. question: how to measure secure?
                possible answer 2: multi-dimensional evaluatation (geo-location of node, history secure weight(error in qubit) current network conditions, historical data, and potential security threats)
        4. return S(P)

    submodule 1 (Routing)
        1. Find n number of path, according to the real-time safety evaluation of each node, path from S(P).

    submodule 2 (Excution)
        1. Generate and send Qubit string through n number of path.

module 2 (Intrusion / Adversary Simulation)
    1. Eve can move in space
    2. Variable: 
        a. pervasiveness: Percentage of nodes occupied by Eve.
        b. capability: Eve's probability of captureing parent string on a single node
    3. Determination of success eavesdropping:
        a. Case 1. At least 1 safe 

Module 3: Security Computation and Performance Evaluation
    Purpose:
        To quantify the probability that an attacker successfully intercepts the distributed quantum key
        when a certain proportion of network nodes are compromised. ASR reflects the security robustness of 
        the multi-path QKD routing under hybrid-trusted conditions.

    Definitions:
        epoch / request: one complete key-distribution process between a source and destination.
        N_total: total number of transmission requests in the simulation.
        N_succ: number of requests where the attacker successfully intercepts the key according to the selected attack model.
        ASR = N_succ / N_total.
    
    Attack Models:
        Case A (Any-Path Compromise): 
            the attack is successful if the attacker fully captures all information on at least one active path between the source and destination.
        Case B (Threshold t-of-n Compromise): 
            the attack is successful if the attacker captures not fewer than t of the n parallel paths used for key transmission. This corresponds to the multi-path threshold key-sharing model described in the main paper.
    
    Implementation Steps (per epoch):
        1. After the Transmission module selects n paths {P1, P2, …, Pn}, the Intrusion module marks a subset of nodes as compromised with probability p and simulates their positions.
        2. For each path Pi, determine whether it is captured (i.e., every relay node on Pi is compromised or the path passes through a compromised segment). Store captured_i = True / False.
        3. Evaluate attack success:
            Case A: success_epoch = any(captured_i == True).
            Case B: success_epoch = (sum(captured_i) ≥ t_threshold).
        4. If success_epoch is True, increment N_succ. Always increment N_total.
        5. After all epochs, compute ASR = N_succ / N_total.

        Analytical Approximation:
        If each path has independent capture probability r, then
            Case A: P_success = 1 - (1 - r)^n.
            Case B: P_success = ∑_{k=t}^{n} C(n,k) r^k (1 - r)^{n-k}.