import collections

class Solution:
  def timeTaken(self, edges: list[list[int]]) -> list[int]:
    n_val = 0
    # Constraints: 2 <= n <= 10^5. edges.length == n - 1.
    # So edges is never empty, and n is at least 2.
    # n is the number of nodes. Nodes are 0 to n-1.
    # The largest node index found in edges determines n when nodes are 0-indexed compactly.
    for u_node, v_node in edges:
        n_val = max(n_val, u_node, v_node)
    n = n_val + 1

    adj = [[] for _ in range(n)]
    for u_node, v_node in edges:
        adj[u_node].append(v_node)
        adj[v_node].append(u_node)

    # w_cost[i] is the cost associated with arriving at node i from an adjacent marked node.
    w_cost = [0] * n
    for i in range(n):
        w_cost[i] = 1 if i % 2 != 0 else 2 # 1 if odd, 2 if even

    # Arbitrarily root the tree at node 0 for DP calculations.
    root = 0
    
    # Step 1: BFS from root.
    # This establishes parent-child relationships (children_map) for the DP.
    # It also gives an ordering of nodes (processing_order_bfs) where parents appear before children.
    processing_order_bfs = [] 
    children_map = [[] for _ in range(n)] # children_map[u] stores children of u in BFS tree

    q_bfs_deque = collections.deque()
    q_bfs_deque.append((root, -1)) # (node, parent_in_bfs_tree)
    
    visited_in_bfs = [False] * n
    visited_in_bfs[root] = True
    
    while q_bfs_deque:
        u, p_of_u = q_bfs_deque.popleft()
        processing_order_bfs.append((u, p_of_u))
        
        for v_neighbor in adj[u]:
            if v_neighbor == p_of_u: # Don't go back up to parent in BFS tree
                continue
            
            if not visited_in_bfs[v_neighbor]: # Process unvisited neighbors
                visited_in_bfs[v_neighbor] = True
                children_map[u].append(v_neighbor)
                q_bfs_deque.append((v_neighbor, u))
    
    # Step 2: Calculate dp_down values.
    # dp_down[u] = max path sum starting at u and going strictly into its BFS subtree.
    # Requires a post-order traversal (children processed before their parent).
    # Iterating processing_order_bfs in reverse achieves this.
    dp_down = [0] * n
    for i in range(n - 1, -1, -1): # Iterate from last node in BFS order to first (root)
        u, _ = processing_order_bfs[i] # Parent p_of_u is not needed for this calculation logic
        # dp_down[u] is initialized to 0 (path sum from u to u itself).
        for v_child in children_map[u]:
            # Path from u to x in v_child's subtree: u -> v_child -> ... -> x
            # Path sum = w_cost[v_child] + (sum for path v_child...x)
            dp_down[u] = max(dp_down[u], w_cost[v_child] + dp_down[v_child])

    # Step 3: Calculate dp_up values and final answers.
    # dp_up[u] = max path sum starting at u and going "upwards" (not into its BFS subtree).
    # Requires a pre-order traversal (parents processed before children).
    # Iterating processing_order_bfs in forward order achieves this.
    dp_up = [0] * n  # dp_up[root] is 0, correctly initialized.
    ans = [0] * n
    
    for i in range(n): # Iterate from root to leaves (BFS order)
        u, p_of_u = processing_order_bfs[i]

        # To calculate dp_up[v_child] for all children v_child of u, we need:
        # 1. dp_up[u]: Max path sum from u, going towards p_of_u (parent of u).
        # 2. max_{s sibling of v_child} (w_cost[s] + dp_down[s]): Max path sum from u, 
        #    going into a sibling's subtree.
        # For efficient calculation of (2), precompute prefix and suffix maximums of
        # (w_cost[c] + dp_down[c]) for all children c of u.
        
        child_contributions = [] # Stores (w_cost[child] + dp_down[child])
        for v_child_for_contrib in children_map[u]:
            child_contributions.append(w_cost[v_child_for_contrib] + dp_down[v_child_for_contrib])
        
        num_children = len(children_map[u])
        
        prefix_max_contribs = [0] * num_children 
        # prefix_max_contribs[j] = max of child_contributions[0...j]
        # Path sums are non-negative. Min w_cost is 1. Min dp_down is 0. So min contribution is 1.
        # Initializing current_max_val = 0 is safe.
        current_max_val = 0 
        if num_children > 0:
            for j in range(num_children):
                current_max_val = max(current_max_val, child_contributions[j])
                prefix_max_contribs[j] = current_max_val
        
        suffix_max_contribs = [0] * num_children
        # suffix_max_contribs[j] = max of child_contributions[j...num_children-1]
        current_max_val = 0
        if num_children > 0:
            for j in range(num_children - 1, -1, -1):
                current_max_val = max(current_max_val, child_contributions[j])
                suffix_max_contribs[j] = current_max_val

        # For each child v_child of u, calculate its dp_up[v_child].
        # A path starting at v_child and going "up" must first go to u. Cost: w_cost[u].
        # Then from u, it can go towards p_of_u or towards a sibling of v_child.
        for j_idx, v_child in enumerate(children_map[u]): # j_idx is index of v_child in children_map[u]
            
            # Max path sum from u, going "upwards" via p_of_u.
            path_val_from_u_upwards = dp_up[u]
            
            # Max path sum from u, going "downwards" into a sibling's subtree.
            path_val_from_u_to_siblings = 0
            if j_idx > 0: # Sibling(s) to the "left" (processed earlier in children_map[u])
                path_val_from_u_to_siblings = max(path_val_from_u_to_siblings, prefix_max_contribs[j_idx-1])
            if j_idx < num_children - 1: # Sibling(s) to the "right"
                path_val_from_u_to_siblings = max(path_val_from_u_to_siblings, suffix_max_contribs[j_idx+1])
            
            dp_up[v_child] = w_cost[u] + max(path_val_from_u_upwards, path_val_from_u_to_siblings)
        
        # The final answer for node u is the maximum of path sums downwards or upwards from u.
        ans[u] = max(dp_down[u], dp_up[u])
            
    return ans