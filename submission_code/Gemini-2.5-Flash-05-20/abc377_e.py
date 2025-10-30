import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    P_orig_1_indexed = list(map(int, sys.stdin.readline().split()))
    
    # Convert to 0-indexed
    P_orig = [p - 1 for p in P_orig_1_indexed]

    final_P = [0] * N # Stores the final permutation values (0-indexed)
    
    # visited[i] = True if final_P[i] has been determined
    visited = [False] * N 

    # _P_orig_props[node] stores (pre_len, cycle_len, cycle_entry_node) for the node's path/cycle
    # This info is needed when a path (tail) merges into an already processed structure.
    _P_orig_props = [None] * N 

    # Sieve for Euler's totient function (phi) up to N
    # phi_sieve[i] will store phi(i)
    phi_sieve = list(range(N + 1))
    for i in range(2, N + 1):
        if phi_sieve[i] == i: # i is prime
            for j in range(i, N + 1, i):
                phi_sieve[j] -= phi_sieve[j] // i

    # Precompute jump table for powers of 2.
    # jump_table[s][node] stores the node that `node` maps to after 2^s applications of P_orig.
    # Max s needed is log2(N) + some buffer (e.g. N.bit_length()).
    MAX_JUMP_S = N.bit_length() # max s such that 2^s <= N. Could be slightly larger (e.g. 20 for N=2e5)
    jump_table = [[0] * N for _ in range(MAX_JUMP_S)]

    # Base case: 2^0 = 1 step
    for i in range(N):
        jump_table[0][i] = P_orig[i]

    # Fill jump table
    for s in range(1, MAX_JUMP_S):
        for i in range(N):
            jump_table[s][i] = jump_table[s-1][jump_table[s-1][i]]

    # Helper function to apply P_orig_func 'steps' times from 'start_node' using binary lifting
    def apply_permutation_steps(start_node, steps):
        current_node = start_node
        for s in range(MAX_JUMP_S):
            if (steps >> s) & 1: # If the s-th bit of steps is 1
                current_node = jump_table[s][current_node]
        return current_node

    # Process each node to determine its final position
    for i in range(N):
        if not visited[i]:
            # Trace path to find cycle and pre-period
            current_path_nodes = []
            current_path_steps = {} # maps node to its step count from 'i'
            
            curr = i
            step = 0
            
            # Trace path until we hit an already visited node
            while not visited[curr]:
                visited[curr] = True # Mark as visited in global array
                current_path_nodes.append(curr)
                current_path_steps[curr] = step
                curr = P_orig[curr]
                step += 1
            
            # At this point, `curr` is the first node encountered that was already visited.
            
            if curr in current_path_steps:
                # Case 1: `curr` is part of the current path being traced (a cycle is found within this path)
                cycle_start_idx = current_path_steps[curr]
                pre_len = cycle_start_idx # steps from 'i' to the cycle entry node
                cycle_len = len(current_path_nodes) - cycle_start_idx
                cycle_entry_node = current_path_nodes[cycle_start_idx]
                
                # Calculate 2^K mod cycle_len using modular exponentiation and Euler's totient theorem
                # a^b % m. If b >= phi(m), a^b = a^(b % phi(m) + phi(m)) % m
                # Here a=2, b=K, m=cycle_len
                
                effective_exponent_mod_L = 0 # Default for cycle_len = 1
                if cycle_len > 0: # Defensive, cycle_len should be >= 1
                    if cycle_len == 1:
                        effective_exponent_mod_L = 0 # Any steps in a 1-node cycle result in 0 steps from start
                    else:
                        phi_val = phi_sieve[cycle_len]
                        # K is 10^18, phi_val is at most N (2e5). So K will always be >= phi_val.
                        exponent_for_pow2 = (K % phi_val) + phi_val
                        effective_exponent_mod_L = pow(2, exponent_for_pow2, cycle_len)
                
                # Assign final values for all nodes in this discovered path (tail + cycle)
                for j in range(len(current_path_nodes)):
                    node = current_path_nodes[j]
                    
                    # 'steps_from_node_to_cycle_entry' is the number of steps it takes from `node` to reach `cycle_entry_node`.
                    steps_from_node_to_cycle_entry = pre_len - j 
                    
                    # Calculate total effective steps from `cycle_entry_node`
                    # This is (2^K - steps_from_node_to_cycle_entry) steps from `cycle_entry_node`.
                    # Modulo `cycle_len`: (effective_exponent_mod_L - (steps_from_node_to_cycle_entry % cycle_len) + cycle_len) % cycle_len
                    actual_offset_in_cycle = (effective_exponent_mod_L - (steps_from_node_to_cycle_entry % cycle_len) + cycle_len) % cycle_len
                    
                    final_P[node] = apply_permutation_steps(cycle_entry_node, actual_offset_in_cycle)
                    _P_orig_props[node] = (j, cycle_len, cycle_entry_node) # store properties for current node (pre_len_from_i, cycle_len, cycle_entry)

            else:
                # Case 2: `curr` was already visited in a previous overall traversal.
                # Its `final_P[curr]` and `_P_orig_props[curr]` should already be computed.
                # The `current_path_nodes` form a tail leading to `curr`.
                
                # Retrieve `curr`'s cycle properties
                target_pre_len, target_cycle_len, target_cycle_entry = _P_orig_props[curr]

                effective_exponent_mod_L_for_target = 0
                if target_cycle_len > 0:
                    if target_cycle_len == 1:
                        effective_exponent_mod_L_for_target = 0
                    else:
                        phi_val = phi_sieve[target_cycle_len]
                        exponent_for_pow2 = (K % phi_val) + phi_val
                        effective_exponent_mod_L_for_target = pow(2, exponent_for_pow2, target_cycle_len)
                
                # Assign final values for nodes in the tail (from end to beginning)
                # `len(current_path_nodes)` is total length of the tail
                for j in range(len(current_path_nodes) - 1, -1, -1):
                    node = current_path_nodes[j]
                    
                    # `steps_from_node_to_curr` is the number of applications of P_orig to go from `node` to `curr`.
                    steps_from_node_to_curr = step - current_path_steps[node]
                    
                    # We want `P_orig^{2^K}(node)`. This is `P_orig^(2^K - steps_from_node_to_curr)(curr)`.
                    # To apply `(2^K - steps_from_node_to_curr)` steps from `curr`,
                    # we first reach `target_cycle_entry` from `curr` in `target_pre_len` steps.
                    # The total steps from `node` to `target_cycle_entry` is `steps_from_node_to_curr + target_pre_len`.
                    # Let this sum be `total_steps_to_entry_from_node`.
                    # Then we need to apply `P_orig` `(2^K - total_steps_to_entry_from_node)` times starting from `target_cycle_entry`.
                    
                    total_steps_to_entry_from_node = steps_from_node_to_curr + target_pre_len

                    actual_offset_in_target_cycle = (effective_exponent_mod_L_for_target - (total_steps_to_entry_from_node % target_cycle_len) + target_cycle_len) % target_cycle_len
                    
                    final_P[node] = apply_permutation_steps(target_cycle_entry, actual_offset_in_target_cycle)
                    _P_orig_props[node] = (total_steps_to_entry_from_node, target_cycle_len, target_cycle_entry)
                    
    # Print result (1-indexed)
    print(*(val + 1 for val in final_P))

solve()