# The provided starter code imports List. Need to add Set and Tuple.
from typing import List, Tuple, Set
from collections import deque, defaultdict

class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        n = len(nums)
        
        # Build undirected adjacency list
        adj = defaultdict(list)
        for u, v, length in edges:
            adj[u].append((v, length))
            adj[v].append((u, length))
            
        # Build rooted tree adjacency list (edges pointing downwards from root 0)
        # This establishes the ancestor-descendant relationship based on root 0.
        adj_rooted = defaultdict(list)
        queue = deque([(0, -1)]) # (node, parent)
        visited_build = {0}
        
        while queue:
            u, parent = queue.popleft()
            for v, length in adj[u]:
                if v not in visited_build:
                    adj_rooted[u].append((v, length))
                    visited_build.add(v)
                    queue.append((v, u))
        
        # global_max_len_info = [max_len, min_nodes]
        # Initialize with path of length 0, 1 node (any single node is a special path).
        # This will be updated by the DFS starting from each node.
        # Initialize with [0, 1] since a single node is a special path.
        global_max_len_info = [0, 1] 

        def update_global_max(current_path_info: Tuple[int, int]):
            nonlocal global_max_len_info
            current_len, current_nodes = current_path_info
            if current_len > global_max_len_info[0]:
                global_max_len_info[0] = current_len
                global_max_len_info[1] = current_nodes
            elif current_len == global_max_len_info[0]:
                global_max_len_info[1] = min(global_max_len_info[1], current_nodes)

        # DFS function to find downward special paths starting from the node where this DFS subtree is rooted (implicitly 'start_node')
        # u: current node in the downward path being explored
        # current_length_to_u: accumulated length of the path from start_node to u
        # values_on_path_excluding_u: set of nums values on the path from start_node up to the parent of u
        # nodes_on_path_excluding_u: number of nodes on the path from start_node up to the parent of u
        
        def dfs_final(u: int, current_length_to_u: int, values_on_path_excluding_u: Set[int], nodes_on_path_excluding_u: int):
            val_u = nums[u]

            # If adding the current node's value violates uniqueness
            if val_u in values_on_path_excluding_u:
                return # This path is not special through u

            # Path up to u is special. Add u's value to the set.
            # Use add/remove on a single set for efficiency
            values_on_path_excluding_u.add(val_u) # visited_values_set now includes u

            # Update global max for the special path ending at u
            # Length is current_length_to_u
            # Number of nodes is nodes_on_path_excluding_u + 1
            update_global_max((current_length_to_u, nodes_on_path_excluding_u + 1))

            # Explore children
            for v, length_uv in adj_rooted[u]:
                # Recurse to child v.
                # length_to_v = length_to_u + length_uv
                # nodes_before_v = nodes_on_path_excluding_u + 1 (nodes up to u)
                # values_on_path_excluding_v = values_on_path_including_u (values up to u)
                dfs_final(v, current_length_to_u + length_uv, values_on_path_excluding_u, nodes_on_path_excluding_u + 1)

            # Backtrack: remove u's value from the set
            values_on_path_excluding_u.remove(val_u)


        # Run DFS starting from every possible node.
        # For each start_node, the path begins.
        # Current node is start_node. Length to start_node is 0.
        # Number of nodes *before* start_node is 0.
        # Set of values *before* start_node is empty.
        for start_node in range(n):
             dfs_final(start_node, 0, set(), 0)


        return global_max_len_info