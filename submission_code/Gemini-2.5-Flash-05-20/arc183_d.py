import sys

# Increase recursion limit for deep trees
sys.setrecursionlimit(10**6)

def solve():
    N = int(sys.stdin.readline())
    
    # adj will store the graph structure of the original tree T.
    # It's not strictly needed for this specific DFS approach on T', but can be useful for debugging or other logic.
    # We store (neighbor_vertex, edge_type) where edge_type is 0 for base edge, 1 for inter-pair edge.
    adj = [[] for _ in range(N + 1)] 
    
    # inter_pair_adj will store the graph structure of T' (tree of pairs).
    # inter_pair_adj[pair_id] will contain tuples (neighbor_pair_id, u_orig, v_orig),
    # where (u_orig, v_orig) is the actual edge in T that connects pair_id to neighbor_pair_id.
    inter_pair_adj = [[] for _ in range(N // 2 + 1)] 
    
    # paired_vertex[v] stores the partner of v in the (2i-1, 2i) pair.
    paired_vertex = [0] * (N + 1)
    
    # pair_id[v] stores the ID of the pair (2i-1, 2i) that v belongs to.
    pair_id = [0] * (N + 1)

    # Process the first N/2 edges, which are (2i-1, 2i)
    for i in range(1, N // 2 + 1):
        u = 2 * i - 1
        v = 2 * i
        adj[u].append((v, 0)) # Type 0 for base edge
        adj[v].append((u, 0))
        paired_vertex[u] = v
        paired_vertex[v] = u
        pair_id[u] = i
        pair_id[v] = i
    
    # Process the remaining N/2 - 1 edges (inter-pair edges)
    for _ in range(N // 2, N - 1): 
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append((v, 1)) # Type 1 for inter-pair edge
        adj[v].append((u, 1))
        
        pair_u_id = pair_id[u]
        pair_v_id = pair_id[v]
        
        # Add edge to T' only if they connect different pairs
        if pair_u_id != pair_v_id:
            # Store which original vertex is part of the connection from this pair_id's side
            inter_pair_adj[pair_u_id].append((pair_v_id, u, v))
            inter_pair_adj[pair_v_id].append((pair_u_id, v, u))

    result_pairs = [] # Stores the (X, Y) pairs to be printed
    visited_T_prime = [False] * (N // 2 + 1) # To keep track of visited nodes in T'
    
    # The DFS on T' returns an exposed vertex (original vertex ID) from its subtree, or None.
    # An exposed vertex is one that could not be paired within its child's subtree or within its own pair.
    # These exposed vertices are then paired with other exposed vertices to maximize distance.
    def dfs_prime(u_prime_id, p_prime_id):
        visited_T_prime[u_prime_id] = True
        
        # stack_for_current_pair stores original vertex IDs that are exposed from children subtrees
        # or from the current pair itself (its 'downward_facing_node').
        # It operates like a stack (LIFO) to encourage pairing "deepest" available nodes.
        stack_for_current_pair = [] 
        
        # Process children in T'
        for v_prime_id, u_orig_conn_v, v_orig_conn_u in inter_pair_adj[u_prime_id]:
            if v_prime_id == p_prime_id or visited_T_prime[v_prime_id]:
                continue
            
            # Recursive call for child's subtree
            exposed_from_child = dfs_prime(v_prime_id, u_prime_id)
            
            # If child subtree has an exposed vertex, add it to current stack
            if exposed_from_child is not None:
                stack_for_current_pair.append(exposed_from_child)
        
        # Determine the 'downward' facing vertex of the current pair u_prime_id.
        # This is the vertex from P_u_prime_id that connects to children in T',
        # or is simply the 'other' vertex if one connects upwards to parent.
        downward_facing_node = None
        
        if p_prime_id is None: # u_prime_id is the root of T'
            # Arbitrarily pick one of the two vertices in this pair to be the "downward" one.
            # The other (2*u_prime_id-1) implicitly acts as the "upward" one for the root.
            downward_facing_node = 2 * u_prime_id 
        else:
            # Find which original vertex within u_prime_id's pair connects to p_prime_id.
            u_up_conn = None 
            for neighbor_pair_id, u_from_curr_pair, v_from_neigh_pair in inter_pair_adj[u_prime_id]:
                if neighbor_pair_id == p_prime_id:
                    u_up_conn = u_from_curr_pair # This is the one connecting upwards to parent.
                    break
            # The other vertex in the pair is the downward facing one.
            downward_facing_node = paired_vertex[u_up_conn] 
        
        # Add the downward_facing_node to the stack.
        # This vertex is now available for pairing with other exposed vertices.
        if downward_facing_node is not None: # Should always be not None for N > 0
            stack_for_current_pair.append(downward_facing_node)
            
        # Pair up as many exposed vertices as possible from the current stack.
        # This greedy pairing strategy typically maximizes total distance.
        while len(stack_for_current_pair) >= 2:
            v1 = stack_for_current_pair.pop()
            v2 = stack_for_current_pair.pop()
            result_pairs.append((v1, v2)) # Record the pair to be removed.
        
        # If one vertex remains, it's exposed and passed up to the parent.
        # This means it couldn't be paired within its subtree and needs a match from higher up.
        if len(stack_for_current_pair) == 1:
            return stack_for_current_pair[0]
        else: # The stack is empty; all vertices in this pair's subtree were matched/removed.
            return None

    # Start DFS from pair_id 1 (arbitrary root for T').
    # The return value from the initial call to dfs_prime(1, None) is ignored,
    # as any remaining exposed node (only possible if N/2 is odd) cannot be paired further.
    # The problem asks for N/2 operations, so it implies N/2 is even for all pairs to be removed.
    # If N/2 is odd, one node will remain unmatched, which aligns with perfect matching on odd number of nodes.
    # However, for this problem, N is even, meaning N/2 is an integer. Thus all nodes are removed.
    dfs_prime(1, None) 
    
    # Print results
    for x, y in result_pairs:
        sys.stdout.write(f"{x} {y}
")

solve()