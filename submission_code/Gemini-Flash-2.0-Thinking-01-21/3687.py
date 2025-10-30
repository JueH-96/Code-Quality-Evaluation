from typing import List
import collections

class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        n = len(nums)

        # Build undirected adjacency list
        adj = [[] for _ in range(n)]
        for u, v, length in edges:
            adj[u].append((v, length))
            adj[v].append((u, length))

        # Build directed adjacency list (parent -> child) starting from root 0
        # We use BFS to traverse the tree from root 0 and establish parent-child relationships.
        directed_adj = [[] for _ in range(n)]
        
        q = collections.deque([(0, -1)]) # (current_node, parent_node)
        visited = {0}
        
        while q:
            u, parent = q.popleft()
            
            for v, length in adj[u]:
                if v != parent and v not in visited:
                    visited.add(v)
                    directed_adj[u].append((v, length))
                    q.append((v, u))
                    
        # Global variables to store the result
        # Initialize with a single node path (length 0, nodes 1), which is always a special path
        max_length = 0
        min_nodes = 1

        # Store the path from root to the current node during DFS.
        # This list acts as a stack representing the current path.
        # Each element is a tuple: (node_index, cumulative_length_from_root_to_this_node)
        path_info = [] 

        def dfs(u, current_len_from_root):
            nonlocal max_length, min_nodes

            # Add current node to the path from root
            path_info.append((u, current_len_from_root))

            # Check all special downward paths ending at the current node 'u'.
            # A special path ending at 'u' starts at some ancestor 'a' on the path from root to u.
            # We iterate upwards from 'u' towards the root along the current path_info.
            seen_values = set()
            
            # Iterate from the current node 'u' (last element in path_info) backwards up the path to the root.
            # path_info currently contains [(root, len_to_root), ..., (ancestor, len_to_ancestor), ..., (u, len_to_u)]
            # The last element path_info[-1] is (u, current_len_from_root).
            for i in range(len(path_info) - 1, -1, -1):
                node_on_path, len_to_node_on_path = path_info[i]
                
                if nums[node_on_path] in seen_values:
                    # The value of node_on_path is already present in the path segment
                    # from the node at index i+1 up to the current node u.
                    # Thus, the path segment from node_on_path up to u is NOT special
                    # because it contains a repeated value (nums[node_on_path]).
                    # Any downward path starting at an ancestor *above* node_on_path (at an index < i)
                    # and ending at u will necessarily include this non-special segment and therefore
                    # also be non-special itself. We can stop checking ancestors further up this path.
                    break
                    
                seen_values.add(nums[node_on_path])
                
                # The path from node_on_path up to u is special (all values encountered so far are unique).
                path_len = current_len_from_root - len_to_node_on_path
                num_nodes = len(path_info) - i # Number of nodes in the path segment from node_on_path to u inclusive

                # Update global maximums based on this special path
                if path_len > max_length:
                    max_length = path_len
                    min_nodes = num_nodes
                elif path_len == max_length:
                    min_nodes = min(min_nodes, num_nodes)

            # Recurse on children in the directed tree
            for v, length in directed_adj[u]:
                # 'v' is guaranteed to be a child of 'u' in the directed tree built from root.
                dfs(v, current_len_from_root + length)

            # Backtrack: Remove the current node 'u' from the path from root
            # as we finish exploring the subtree rooted at 'u'.
            path_info.pop()

        # Start DFS from the root node (node 0) with initial length 0.
        dfs(0, 0) 

        # Return the results stored in the global variables.
        return [max_length, min_nodes]