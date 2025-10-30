import sys

# Increase recursion limit for potentially deep DFS paths in cycle detection
sys.setrecursionlimit(2025 + 500) # N + buffer

MOD = 998244353

def solve():
    N, M = map(int, sys.stdin.readline().split())
    # A is 1-indexed in problem, converting to 0-indexed for array access
    A = [x - 1 for x in map(int, sys.stdin.readline().split())]

    # Precompute factorials and inverse factorials for nCr calculation
    # Max n for nCr(n, r) is k + depth[i] - 1. Max k=M, max depth[i]=N-1.
    # So max n is M + (N-1) - 1 = M + N - 2.
    # Precomputing up to M + N is sufficient.
    MAX_NCR_N = M + N
    fact = [1] * (MAX_NCR_N + 1)
    invfact = [1] * (MAX_NCR_N + 1)

    for i in range(1, MAX_NCR_N + 1):
        fact[i] = (fact[i-1] * i) % MOD

    # invfact[k] = (k!)^(-1) mod MOD
    invfact[MAX_NCR_N] = pow(fact[MAX_NCR_N], MOD - 2, MOD) # Fermat's Little Theorem for modular inverse
    for i in range(MAX_NCR_N - 1, -1, -1):
        invfact[i] = (invfact[i+1] * (i+1)) % MOD

    def nCr_mod_p(n, r):
        if r < 0 or r > n:
            return 0
        return (((fact[n] * invfact[r]) % MOD) * invfact[n-r]) % MOD

    # Step 1: Find cycles and depths for all nodes
    # state[i]: 0 = unvisited, 1 = visiting (on current DFS path), 2 = visited (fully processed)
    state = [0] * N 
    depth = [-1] * N # depth[i] = distance from i to its cycle. 0 for cycle nodes.
    cycle_id = [-1] * N # Which cycle component node i belongs to.
    num_cycles = 0

    # Iterate through each node to ensure all components are processed
    for i in range(N):
        if state[i] == 0:
            curr = i
            path = [] # Stores nodes on the current DFS path for cycle detection
            while state[curr] == 0:
                state[curr] = 1 # Mark as visiting
                path.append(curr)
                curr = A[curr] # Move to the next node according to A

            if state[curr] == 1: # Cycle detected: 'curr' is already on the current path
                num_cycles += 1
                start_of_cycle_idx = path.index(curr)
                
                # Nodes from start_of_cycle_idx to the end of 'path' form the cycle
                for node_in_cycle in path[start_of_cycle_idx:]:
                    state[node_in_cycle] = 2 # Mark as visited
                    depth[node_in_cycle] = 0 # Cycle nodes have depth 0
                    cycle_id[node_in_cycle] = num_cycles # Assign common cycle ID
                
                # Nodes before the cycle in 'path' are non-cycle nodes (trees leading to the cycle)
                d = 1
                for node_in_tree in reversed(path[:start_of_cycle_idx]):
                    state[node_in_tree] = 2
                    depth[node_in_tree] = d
                    cycle_id[node_in_tree] = num_cycles # These also belong to the same component
                    d += 1
            else: # state[curr] == 2: 'curr' has already been fully processed (part of another component)
                # The 'path' consists of nodes leading to 'curr'
                # Assign depths and cycle_ids for nodes in 'path' in reverse order
                for node_in_tree in reversed(path):
                    state[node_in_tree] = 2
                    # The depth and cycle_id of A[node_in_tree] are already known
                    depth[node_in_tree] = depth[A[node_in_tree]] + 1
                    cycle_id[node_in_tree] = cycle_id[A[node_in_tree]]

    # Step 2: Group non-cycle nodes by their assigned cycle_id
    # group_by_cycle[c_id] will contain a list of depths for nodes that lead to cycle c_id
    from collections import defaultdict
    group_by_cycle = defaultdict(list)
    for i in range(N):
        if depth[i] > 0: # Only consider non-cycle nodes
            group_by_cycle[cycle_id[i]].append(depth[i])

    # Step 3: Calculate the total number of ways
    total_ans = 1
    # Iterate over each distinct cycle component identified
    for c_id in range(1, num_cycles + 1):
        current_cycle_component_sum = 0
        
        # Iterate over all possible values 'k' (from 1 to M) for the cycle nodes X_C
        for k in range(1, M + 1):
            current_product_for_k = 1
            
            # Multiply terms for each non-cycle node 'i' belonging to this component
            # Each term is C(k + depth[i] - 1, depth[i])
            for d in group_by_cycle[c_id]:
                current_product_for_k = (current_product_for_k * nCr_mod_p(k + d - 1, d)) % MOD
            
            current_cycle_component_sum = (current_cycle_component_sum + current_product_for_k) % MOD
        
        # Multiply the sum for the current cycle component into the total answer
        total_ans = (total_ans * current_cycle_component_sum) % MOD
    
    print(total_ans)

solve()