import math

def solve():
    N = int(input())
    P_orig = list(map(int, input().split()))
    A_orig = list(map(int, input().split()))

    # Adjust to 0-indexed P, A remains 1-indexed values but in 0-indexed list
    P_py = [x - 1 for x in P_orig] # P_py[i] is where node i gets mapped to
    A_py = A_orig # A_py[i] is the value at original position i+1. Values 1 to N.

    # Decompose P_py into cycles
    visited_node = [False] * N
    # node_to_cycle_info stores: node_idx -> (cycle_list_ref, pos_in_cycle, cycle_len)
    node_to_cycle_info = {} 

    for i in range(N):
        if not visited_node[i]:
            current_cycle_nodes = []
            curr = i
            while not visited_node[curr]:
                visited_node[curr] = True
                current_cycle_nodes.append(curr)
                curr = P_py[curr]
            
            cycle_len = len(current_cycle_nodes)
            for k_enum, node_idx_in_cycle in enumerate(current_cycle_nodes):
                # Store reference to list, pos, len
                node_to_cycle_info[node_idx_in_cycle] = (current_cycle_nodes, k_enum, cycle_len)

    k_cur = 0
    M_cur = 1

    for j_perm_idx in range(N): # Iterate through indices 0 to N-1 of the permutation A
        
        # For current position j_perm_idx in A, what is P^k[j_perm_idx]?
        # This P^k[j_perm_idx] is an index into A_py.
        # A_py[ P^k[j_perm_idx] ] is the value we want to minimize.
        
        # Info for node j_perm_idx
        # cycle_list: the actual list of nodes e.g. [0,2,4,1]
        # pos_in_cycle: initial position of j_perm_idx in its cycle_list
        # cycle_len: length of this cycle_list
        cycle_list, pos_in_cycle, cycle_len = node_to_cycle_info[j_perm_idx]

        # Candidates for k are: k_cur_old + t * M_cur_old
        # (P_py)^k[j_perm_idx] = cycle_list [ (pos_in_cycle + k) % cycle_len ]
        #                       = cycle_list [ (pos_in_cycle + k_cur_old + t*M_cur_old) % cycle_len ]
        
        # Projection of k_cur onto current cycle for j_perm_idx
        k_offset_in_cycle = (pos_in_cycle + k_cur) % cycle_len
        # Projection of M_cur step onto current cycle for j_perm_idx
        M_cur_proj_in_cycle = M_cur % cycle_len
        
        # Number of distinct values (P_py)^k[j_perm_idx] can take for current candidates for k
        # This is p_val = cycle_len / gcd(M_cur_proj_in_cycle, cycle_len)
        common_divisor = math.gcd(M_cur_proj_in_cycle, cycle_len)
        p_val = cycle_len // common_divisor
        
        min_A_val_for_Pkj = float('inf')
        s_best = 0 # Default to 0 if p_val is 1.

        if p_val > 1: # If p_val is 1, (P_py)^k[j_perm_idx] is fixed, s_best remains 0.
            for s_iter in range(p_val):
                # Index in cycle_list for (P_py)^k[j_perm_idx]
                idx_in_cl = (k_offset_in_cycle + s_iter * M_cur_proj_in_cycle) % cycle_len
                
                # The node index obtained after applying P^k transformation
                actual_node_idx_for_A_lookup = cycle_list[idx_in_cl]
                # Value from A_py at this transformed index
                current_A_val = A_py[actual_node_idx_for_A_lookup]

                if current_A_val < min_A_val_for_Pkj:
                    min_A_val_for_Pkj = current_A_val
                    s_best = s_iter
        
        M_cur_old = M_cur
        # Update k_cur to be the smallest non-negative k in the new refined set of candidates
        k_cur = (k_cur + s_best * M_cur_old) 
        
        # Update M_cur, effectively M_cur = lcm(M_cur_old, period_of_choices_for_this_j)
        # where period_of_choices_for_this_j = M_cur_old * p_val
        M_cur = M_cur_old * p_val
        k_cur %= M_cur # Keep k_cur minimal non-negative in [0, M_cur-1]
    
    # Construct the result using final k_cur
    # ans_perm[i] = A_py [ (P_py)^k_cur [i] ]
    ans_perm_values = [0] * N
    for i in range(N): # For each position i in the final permutation
        cycle_list, pos_in_cycle, cycle_len = node_to_cycle_info[i]
        
        # Find (P_py)^k_cur [i]
        idx_in_cl_for_Pk_i = (pos_in_cycle + k_cur) % cycle_len
        actual_node_idx_for_A_lookup = cycle_list[idx_in_cl_for_Pk_i]
        
        ans_perm_values[i] = A_py[actual_node_idx_for_A_lookup]
        
    print(*(ans_perm_values))

solve()