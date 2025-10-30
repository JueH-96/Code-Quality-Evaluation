from collections import deque
from typing import List

class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        n = len(nums)
        # Find the maximum value in nums to determine the size of the visited_values array
        max_val = max(nums)
        
        # Build undirected graph
        adj_undirected = [[] for _ in range(n)]
        for u, v, length in edges:
            adj_undirected[u].append((v, length))
            adj_undirected[v].append((u, length))
        
        # Build parent pointers and edge length to parent from root 0 using BFS
        parent = [-1] * n
        up_edge_len = [0] * n # Stores length from node to parent
        q = deque([(0, -1, 0)]) # (node, parent, length_to_parent)
        
        # In a tree, visited check is often implicit when avoiding the parent,
        # but explicitly marking visited can prevent cycles in general graphs.
        # For a tree, avoiding the parent is sufficient.
        
        visited_build = [False] * n
        visited_build[0] = True # Mark root as visited

        q = deque([0]) # Queue stores nodes to visit
        parent[0] = -1 # Root has no parent
        # up_edge_len[0] remains 0, or some indicator it's not applicable

        while q:
            u = q.popleft()
            
            for v, length in adj_undirected[u]:
                if v != parent[u]: # If v is not the parent of u
                    parent[v] = u
                    up_edge_len[v] = length # Edge length from v to parent[v]
                    q.append(v)

        # Global result variables
        max_length = 0
        min_nodes = 1 # A single node path has length 0, nodes 1

        # Iterate through each node as the potential END node of a special path
        for end_node in range(n):
            
            # visited_values will store values encountered on the path
            # from the current ancestor (curr_upward) down to end_node.
            # Reset for each new end_node.
            visited_values_on_path = [False] * (max_val + 1)

            curr_upward = end_node # Start traversing upwards from the end node
            current_length = 0     # Length of the path from curr_upward down to end_node
            num_nodes_path = 1     # Number of nodes in the path from curr_upward down to end_node
            
            # Traverse upwards from end_node towards the root
            while curr_upward != -1:
                
                val_curr = nums[curr_upward]
                
                # Check if the value at the current node `curr_upward` is already in the path
                # from the node just below it (`prev_upward`) down to `end_node`.
                # `visited_values_on_path` holds values from `prev_upward` down to `end_node`
                # if `curr_upward` is not `end_node`.
                # If `curr_upward` is `end_node` (first iteration), visited_values_on_path is empty initially.
                
                # The condition `if visited_values_on_path[val_curr]` correctly checks if the value
                # of the current ancestor `curr_upward` is already present in the segment
                # of the path *below* it, going down to `end_node`.
                
                if visited_values_on_path[val_curr]:
                     # The path from `curr_upward` down to `end_node` is NOT special because `val_curr` is repeated.
                     # Stop traversing further up for this `end_node`.
                     break 

                # The value `nums[curr_upward]` is unique in the path being considered (`curr_upward` down to `end_node`).
                # Mark the value as visited for the path starting at `curr_upward`.
                visited_values_on_path[val_curr] = True
                
                # This path (from `curr_upward` down to `end_node`) is a special path.
                # Update global result.
                if current_length > max_length:
                    max_length = current_length
                    min_nodes = num_nodes_path
                elif current_length == max_length:
                    min_nodes = min(min_nodes, num_nodes_path)
                
                # Store the current node before moving upwards, to calculate edge length in the next step
                prev_upward = curr_upward 
                
                # Move upwards to the parent node
                curr_upward = parent[curr_upward] 

                # If we successfully moved to a parent, update length and node count
                if curr_upward != -1:
                    # The edge connecting `curr_upward` (the parent) and `prev_upward` (the child)
                    # has length `up_edge_len[prev_upward]`.
                    # This edge is added when we extend the path upwards from `prev_upward` to include `curr_upward`.
                    edge_len_added = up_edge_len[prev_upward]
                    current_length += edge_len_added
                    num_nodes_path += 1

        return [max_length, min_nodes]