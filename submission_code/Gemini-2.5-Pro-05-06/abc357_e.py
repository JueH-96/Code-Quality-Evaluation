import sys

def solve():
    N = int(sys.stdin.readline())
    a_input = list(map(int, sys.stdin.readline().split()))

    # Adjust to 0-indexed adjacency list
    # adj[i] is the 0-indexed vertex that vertex i (0-indexed) points to
    adj = [x - 1 for x in a_input]

    # memo[i] will store a tuple: (distance_to_cycle, cycle_length) for vertex i.
    # distance_to_cycle is the number of edges on the path from i to the first vertex in a cycle.
    # If i is in a cycle, distance_to_cycle is 0.
    # cycle_length is the length of the cycle reached from i.
    memo = [None] * N 
    
    total_reachable_pairs = 0

    for i in range(N):
        if memo[i] is not None:
            # This vertex has already been processed as part of another path.
            # Its contribution to total_reachable_pairs was already added when it was processed.
            continue

        # Trace the path starting from vertex i to find a cycle or an already processed node.
        current_path_nodes = []  # Stores nodes in the current trace, in order.
        
        # path_nodes_to_steps maps a node index to its step number (0-indexed position)
        # in the current_path_nodes list for the current trace.
        path_nodes_to_steps = {} 
        
        curr = i
        # current_step_count will be the length of current_path_nodes when the loop ends.
        # It also represents the number of edges traversed from the start of the trace (node i)
        # to reach the predecessor of the current `curr` node.
        current_step_count = 0 
        
        while memo[curr] is None and curr not in path_nodes_to_steps:
            path_nodes_to_steps[curr] = current_step_count
            current_path_nodes.append(curr)
            curr = adj[curr]
            current_step_count += 1
        
        # Path tracing stops. `curr` is the node that caused termination.
        # `current_path_nodes` contains [p_0, p_1, ..., p_{k-1}], where p_0 = i.
        # `k = len(current_path_nodes) = current_step_count` (value of current_step_count after loop).
        # `curr` is the successor of p_{k-1}, i.e., adj[p_{k-1}].

        if memo[curr] is not None:
            # Path merges into an already processed component.
            # `curr` is the first node encountered that is already in `memo`.
            base_dist_to_cycle, cycle_len = memo[curr]
            
            # Iterate current_path_nodes in reverse to fill memo entries.
            # For a node p_j = current_path_nodes[j]:
            # Its distance to `curr` (the junction node) is (k-1-j) + 1 = k-j edges.
            # `k` is `current_step_count`. `j` is `path_nodes_to_steps[p_j]`.
            # So, distance from p_j to `curr` = current_step_count - path_nodes_to_steps[p_j].
            
            # Iterate from the node just before the junction point, back to the start of the current path.
            for node_idx_in_path_trace in range(len(current_path_nodes) - 1, -1, -1):
                node_on_this_path = current_path_nodes[node_idx_in_path_trace]
                
                # Distance from node_on_this_path to `curr` (the junction node).
                dist_to_junction_node = current_step_count - path_nodes_to_steps[node_on_this_path]

                # Total distance to cycle for node_on_this_path
                dist = dist_to_junction_node + base_dist_to_cycle
                # Cycle length remains the same as the one reached by `curr`.
                len_of_cycle_reached = cycle_len
                
                memo[node_on_this_path] = (dist, len_of_cycle_reached)
                # Number of reachable nodes from node_on_this_path is dist + len_of_cycle_reached
                total_reachable_pairs += dist + len_of_cycle_reached
        
        else: 
            # A cycle is detected because `curr` is already in `path_nodes_to_steps`.
            # This means `curr` is one of the nodes in `current_path_nodes`.
            
            # Find where the cycle begins in `current_path_nodes`.
            # `path_nodes_to_steps[curr]` gives the index of `curr` in `current_path_nodes`.
            cycle_start_step_idx = path_nodes_to_steps[curr]
            
            # Length of the cycle.
            # All nodes from current_path_nodes[cycle_start_step_idx] to end form the cycle.
            L_cycle = len(current_path_nodes) - cycle_start_step_idx
            # This is also current_step_count - cycle_start_step_idx

            # Process nodes that are part of the cycle.
            # Their distance_to_cycle is 0.
            for k_cycle_node_list_idx in range(cycle_start_step_idx, len(current_path_nodes)):
                node_in_cycle = current_path_nodes[k_cycle_node_list_idx]
                memo[node_in_cycle] = (0, L_cycle)
                total_reachable_pairs += L_cycle # (0 for dist_to_cycle + L_cycle)
            
            # Process nodes on the "stem" path leading into this cycle.
            # These are current_path_nodes[0] up to current_path_nodes[cycle_start_step_idx - 1].
            # Iterate in reverse from the node just before the cycle starts, back to the beginning of the path.
            for k_path_node_list_idx in range(cycle_start_step_idx - 1, -1, -1):
                node_on_stem = current_path_nodes[k_path_node_list_idx]
                
                # Distance from node_on_stem to the start of the cycle.
                # If node_on_stem is current_path_nodes[j] (where j = k_path_node_list_idx),
                # its step recorded in path_nodes_to_steps is j.
                # dist = cycle_start_step_idx - j.
                dist = cycle_start_step_idx - path_nodes_to_steps[node_on_stem]
                
                memo[node_on_stem] = (dist, L_cycle)
                total_reachable_pairs += dist + L_cycle
                
    sys.stdout.write(str(total_reachable_pairs) + "
")

solve()