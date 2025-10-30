import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A_orig = list(map(int, sys.stdin.readline().split()))
    A = [a - 1 for a in A_orig] # 0-indexed

    MOD = 998244353

    on_cycle = [False] * N
    # cycle_id[i] = id of cycle if node i is ON the cycle, else -1
    cycle_id = [-1] * N 
    # component_cycle_id[i] = id of cycle that node i eventually reaches
    component_cycle_id = [-1] * N 
    
    visited_dfs_globally = [False] * N 
    num_cycles = 0

    for i in range(N):
        if visited_dfs_globally[i]:
            continue
        
        current_path_nodes = []
        # current_path_flags stores node -> its index in current_path_nodes
        current_path_flags = {} 
        curr = i

        # Trace path until a node is re-encountered or a globally visited node is hit
        while not visited_dfs_globally[curr] and curr not in current_path_flags:
            current_path_flags[curr] = len(current_path_nodes)
            current_path_nodes.append(curr)
            curr = A[curr]
        
        if curr in current_path_flags: 
            # Cycle detected within current_path_nodes
            new_cycle_unique_id = num_cycles
            
            cycle_start_idx_in_path = current_path_flags[curr]
            # Mark nodes on the new cycle
            for k_path_idx in range(cycle_start_idx_in_path, len(current_path_nodes)):
                node_in_cycle = current_path_nodes[k_path_idx]
                on_cycle[node_in_cycle] = True
                cycle_id[node_in_cycle] = new_cycle_unique_id
                component_cycle_id[node_in_cycle] = new_cycle_unique_id
                visited_dfs_globally[node_in_cycle] = True
            
            # Mark nodes in current_path_nodes that lead into this cycle
            for k_path_idx in range(cycle_start_idx_in_path):
                node_on_path = current_path_nodes[k_path_idx]
                visited_dfs_globally[node_on_path] = True
                component_cycle_id[node_on_path] = new_cycle_unique_id
            
            num_cycles += 1
        elif visited_dfs_globally[curr]:
            # Path leads to an already processed component
            target_cycle_for_path = component_cycle_id[curr]
            for node_on_path in current_path_nodes:
                visited_dfs_globally[node_on_path] = True
                component_cycle_id[node_on_path] = target_cycle_for_path
        # else: This case should not be reached for N >= 1

    V_path_nodes = [] # List of nodes not on any cycle
    for i in range(N):
        if not on_cycle[i]:
            V_path_nodes.append(i)

    # adj_rev[u] = list of nodes v such that A[v] = u
    adj_rev = [[] for _ in range(N)] 
    for node_idx in range(N):
        parent_of_node = A[node_idx]
        adj_rev[parent_of_node].append(node_idx)

    # For topological sort of G_path (where edges are u -> A[u] for u in V_path)
    # We process leaves of G_path first.
    # children_in_G_path_count[u] = number of v in V_path_nodes s.t. A[v]=u.
    # (i.e. u is parent of v in G_path)
    children_in_G_path_count = [0] * N 
    for u_node in V_path_nodes:
        for v_child_candidate in adj_rev[u_node]: 
             if not on_cycle[v_child_candidate]: # Ensure child is in V_path_nodes
                children_in_G_path_count[u_node] += 1
    
    # Queue for topological sort (nodes with 0 children_in_G_path_count are leaves of G_path)
    queue = []
    for u_node in V_path_nodes:
        if children_in_G_path_count[u_node] == 0: 
            queue.append(u_node)
            
    # dp_val_for_x_u[u][k] = product of N_sum_val for children of u, if x_u = k
    dp_val_for_x_u = [[0]*(M + 1) for _ in range(N)] 
    # N_sum_val[u][limit] = N(u, limit) from problem description
    N_sum_val = [[0]*(M + 1) for _ in range(N)]

    processed_idx_in_queue = 0
    while processed_idx_in_queue < len(queue):
        u_curr_processing = queue[processed_idx_in_queue]
        processed_idx_in_queue += 1

        # Calculate dp_val_for_x_u[u_curr_processing][k_val_for_u_curr]
        for k_val_for_u_curr in range(1, M + 1):
            term_prod = 1
            # Children of u_curr_processing in G_rev are v s.t. A[v]=u_curr_processing
            for v_child_of_u_in_Grev in adj_rev[u_curr_processing]:
                if not on_cycle[v_child_of_u_in_Grev]: # v is in V_path_nodes
                    term_prod = (term_prod * N_sum_val[v_child_of_u_in_Grev][k_val_for_u_curr]) % MOD
            dp_val_for_x_u[u_curr_processing][k_val_for_u_curr] = term_prod
        
        # Calculate N_sum_val[u_curr_processing] using prefix sums
        for k_limit_val in range(1, M + 1):
            N_sum_val[u_curr_processing][k_limit_val] = \
                (N_sum_val[u_curr_processing][k_limit_val-1] + \
                 dp_val_for_x_u[u_curr_processing][k_limit_val]) % MOD
            
        # For topological sort: update parent's children count
        parent_of_u_curr = A[u_curr_processing]
        if not on_cycle[parent_of_u_curr]: # If parent is in V_path_nodes
            children_in_G_path_count[parent_of_u_curr] -= 1
            if children_in_G_path_count[parent_of_u_curr] == 0:
                queue.append(parent_of_u_curr)
    
    # Calculate S_j for each cycle
    S_j_cycle_sums = [0] * num_cycles 
    
    # roots_for_each_cycle[j] = list of nodes u in V_path such that A[u] is on cycle j
    roots_for_each_cycle = [[] for _ in range(num_cycles)]
    for u_node in V_path_nodes:
        parent_of_u = A[u_node]
        if on_cycle[parent_of_u]:
            c_id_of_parent = cycle_id[parent_of_u] 
            roots_for_each_cycle[c_id_of_parent].append(u_node)

    for c_idx in range(num_cycles):
        current_S_j_val = 0
        for k_val_for_cycle_nodes in range(1, M + 1): 
            term_prod_F_for_k = 1
            for u_root_node in roots_for_each_cycle[c_idx]:
                term_prod_F_for_k = (term_prod_F_for_k * N_sum_val[u_root_node][k_val_for_cycle_nodes]) % MOD
            current_S_j_val = (current_S_j_val + term_prod_F_for_k) % MOD
        S_j_cycle_sums[c_idx] = current_S_j_val

    ans = 1
    # If num_cycles is 0, this loop won't run, ans remains 1. This is correct if N=0.
    # But N>=1 implies at least one cycle. So num_cycles >= 1.
    for val in S_j_cycle_sums:
        ans = (ans * val) % MOD
        
    print(ans)

solve()