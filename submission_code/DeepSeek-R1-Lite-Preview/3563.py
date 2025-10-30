class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        from collections import defaultdict

        # Collect unique values and assign indices
        unique_values = sorted(set(cell for row in grid for cell in row))
        value_to_index = {value: idx for idx, value in enumerate(unique_values)}
        m = len(grid)
        k = len(unique_values)
        
        # Create adjacency list for bipartite graph
        graph = [[] for _ in range(m)]
        for row_idx, row in enumerate(grid):
            for cell in row:
                val_idx = value_to_index[cell]
                graph[row_idx].append(val_idx)
        
        # Create weight matrix
        weight = [[0] * k for _ in range(m)]
        for row_idx, row in enumerate(grid):
            for cell in row:
                val_idx = value_to_index[cell]
                weight[row_idx][val_idx] = cell  # Assign the value as weight
        
        # Hungarian Algorithm for maximum weight matching in bipartite graph
        def hungarian():
            n_left = m
            n_right = k
            match_to = [ -1 ] * n_right
            # Slack variables for right nodes
            slack = [0.0] * n_right
            # Parent array for BFS
            parent = [0] * n_right
            # Whether a node is in the current tree
            in_tree = [False] * (n_left + n_right)
            
            # For each left node, try to find a matching
            for u in range(n_left):
                # Initialize slack
                slack[:] = weight[u]
                parent[:] = [u] * n_right
                in_tree[:] = [False] * (n_left + n_right)
                
                # Start BFS
                while True:
                    # Find the minimum slack
                    min_slack = float('inf')
                    v_selected = -1
                    for v in range(n_right):
                        if not in_tree[v] and slack[v] < min_slack:
                            min_slack = slack[v]
                            v_selected = v
                    if v_selected == -1:
                        break
                    # Add v_selected to the tree
                    in_tree[v_selected] = True
                    # Find the augmenting path
                    while True:
                        u_temp = parent[v_selected]
                        v_alt = -1
                        for v in range(n_right):
                            if not in_tree[v] and weight[u_temp][v] - match_to[v] if match_to[v] != -1 else weight[u_temp][v] < slack[v] + min_slack:
                                slack[v] = weight[u_temp][v] - match_to[v] if match_to[v] != -1 else weight[u_temp][v]
                                parent[v] = u_temp
                        # Update slack for all non-tree right nodes
                        for v in range(n_right):
                            if not in_tree[v]:
                                slack[v] -= min_slack
                    # Update the matching
                    while True:
                        v_temp = match_to[u_temp]
                        match_to[u_temp] = v_selected
                        if v_temp == -1:
                            break
                        u_temp = parent[v_temp]
                        v_selected = v_temp
            # Calculate the total weight
            total_weight = 0
            for v in range(n_right):
                if match_to[v] != -1:
                    total_weight += weight[match_to[v]][v]
            return total_weight
        
        max_sum = hungarian()
        return max_sum