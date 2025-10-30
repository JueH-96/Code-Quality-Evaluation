import sys

sys.setrecursionlimit(5 * 10**5) # Increased recursion limit for N up to 2.5*10^5

def solve():
    N = int(sys.stdin.readline())
    
    adj = [[] for _ in range(N + 1)]
    
    # P[v] stores the M-partner of v
    P = [0] * (N + 1)
    for i in range(1, N // 2 + 1):
        u_m, v_m = 2 * i - 1, 2 * i
        P[u_m] = v_m
        P[v_m] = u_m

    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    ans_map = [0] * (N + 1) # ans_map[v] will store the vertex v is paired with for removal
    
    parent_in_dfs_tree = [0] * (N + 1)
    order_for_ans_map_definition = [] # Store nodes in post-order
    
    # DFS to populate parent_in_dfs_tree and order_for_ans_map_definition (post-order)
    # Stack for iterative DFS: (vertex, parent_for_dfs_call, state)
    # state=0: just visited, need to explore children
    # state=1: children explored, ready for post-order processing
    
    # Using recursion for simplicity as limit is increased
    # visited_dfs array is not strictly needed for a tree if we just check against parent
    # but good practice if multiple disconnected components were possible (not here)
    
    # DFS to build parent array and get post-order traversal
    def build_dfs_properties(u_dfs, p_dfs):
        parent_in_dfs_tree[u_dfs] = p_dfs
        for v_neighbor in adj[u_dfs]:
            if v_neighbor != p_dfs:
                build_dfs_properties(v_neighbor, u_dfs)
        order_for_ans_map_definition.append(u_dfs)

    if N > 0: # Check if N is positive before calling DFS
        build_dfs_properties(1, 0) # Root DFS tree at vertex 1, parent of root is 0

    # Define ans_map using the post-order traversal
    for u_curr in order_for_ans_map_definition:
        if ans_map[u_curr] == 0: # If u_curr is not yet paired
            p_of_u_curr = parent_in_dfs_tree[u_curr]
            if p_of_u_curr == 0: # u_curr is the root of DFS tree
                ans_map[u_curr] = P[u_curr]
                if P[u_curr] != 0 : # Ensure M-partner exists (always for valid N)
                    ans_map[P[u_curr]] = u_curr
            else:
                ans_map[u_curr] = P[p_of_u_curr]
                if P[p_of_u_curr] != 0:
                     ans_map[P[p_of_u_curr]] = u_curr
                
                ans_map[P[u_curr]] = p_of_u_curr
                if p_of_u_curr != 0: # Parent exists
                    ans_map[p_of_u_curr] = P[u_curr]
    
    current_deg = [0] * (N + 1)
    is_removed = [False] * (N + 1)
    
    for i in range(1, N + 1):
        current_deg[i] = len(adj[i]) # Full degree initially
            
    leaf_q = []
    for i in range(1, N + 1):
        if current_deg[i] == 1: # Check for N=0 if graph can be empty
            leaf_q.append(i)
    
    final_ordered_pairs = []
    head_q = 0
    
    processed_pairs_count = 0
    while processed_pairs_count < N // 2 : # Loop N/2 times for N/2 pairs
        if head_q >= len(leaf_q): 
            # This might indicate an issue or all remaining leaves are already handled.
            # For this problem, it should always find a leaf.
            break 
            
        u_leaf = leaf_q[head_q]
        head_q += 1
        
        if is_removed[u_leaf]:
            continue
            
        v_partner = ans_map[u_leaf]
        
        # One of u_leaf or v_partner must be a leaf. The other must also become a leaf or be one.
        # The logic is that `u_leaf` is a leaf. Its partner `v_partner` must also be removable.
        # This strategy ensures `v_partner` is also effectively a leaf or becomes one allowing removal.
        # It is critical that `ans_map` logic and removal order via `leaf_q` is sound.
        
        final_ordered_pairs.append((u_leaf, v_partner))
        processed_pairs_count += 1
        
        is_removed[u_leaf] = True
        is_removed[v_partner] = True # v_partner might not be u_leaf's neighbor
        
        # Vertices to update neighbors for: u_leaf and v_partner
        nodes_to_update_around = [u_leaf, v_partner]

        for node_center in nodes_to_update_around:
            for neighbor_of_node_center in adj[node_center]:
                if not is_removed[neighbor_of_node_center]: 
                    current_deg[neighbor_of_node_center] -= 1
                    if current_deg[neighbor_of_node_center] == 1:
                        leaf_q.append(neighbor_of_node_center)

    for pair_item in final_ordered_pairs:
        print(pair_item[0], pair_item[1])

solve()