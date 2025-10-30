import collections
import sys
from typing import List

# Setting a high recursion depth limit might be necessary for deep trees, as N can be up to 5*10^4.
# Standard Python limits (often around 1000) might be insufficient.
# LeetCode's Python environment sometimes has a higher limit, but setting it explicitly can be safer.
try:
    # Increase recursion depth limit. Add a buffer over the maximum possible N.
    # Example: Set to 60000 if N max is 50000.
    sys.setrecursionlimit(50000 + 100) 
except Exception as e:
    # Ignore if setting recursion limit fails (e.g., due to environment restrictions).
    # This might happen in certain execution environments.
    pass 

class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        """
        Finds the longest special path in a rooted tree and the minimum number of nodes among such paths.
        A special path is a downward path from an ancestor to a descendant with unique node values.

        Args:
            edges: A list of edges, where edges[i] = [u_i, v_i, length_i].
            nums: A list where nums[i] is the value at node i.

        Returns:
            A list [max_length, min_nodes], where max_length is the length of the longest
            special path, and min_nodes is the minimum number of nodes among all paths
            achieving this maximum length.
        """
        
        n = len(nums)
        # According to constraints, n >= 2, so we don't need to handle n=0 or n=1 cases.

        # Build the adjacency list representation of the undirected tree.
        # Each entry adj[u] is a list of tuples (v, length), representing edges connected to u.
        adj = collections.defaultdict(list)
        for u, v, length in edges:
            adj[u].append((v, length))
            adj[v].append((u, length))

        # Instance variables to store the final result. These are updated during the DFS traversal.
        # self.global_max_len stores the maximum length found so far for any special path.
        self.global_max_len = 0
        # self.global_min_nodes stores the minimum number of nodes among all special paths
        # that achieve the global_max_len. Initialize to 1, as a single node path (length 0) exists.
        self.global_min_nodes = 1 

        # Path tracking structures used during DFS. These are modified during traversal and restored during backtracking.
        # self.path_lengths[d] stores the total path length from the root (node 0) to the node at depth d in the current DFS path.
        self.path_lengths = [0] * n 
        # self.path_values_last_occurrence maps a node value to the maximum depth at which it appeared on the current path from the root.
        # This map is essential for quickly finding the start node of the longest special path ending at the current node.
        self.path_values_last_occurrence = {} 

        def dfs(u: int, parent: int, depth: int, current_length: int):
            """
            Performs Depth First Search to find the longest special paths.

            Args:
                u: The current node being visited.
                parent: The parent node of u in the DFS traversal (to avoid going back up).
                depth: The depth of node u from the root (root node 0 is at depth 0).
                current_length: The total length of the path from the root node 0 to node u.
            """
            
            # Record the path length accumulated till the current node u at its depth.
            # This information is used to calculate path lengths ending at descendants.
            self.path_lengths[depth] = current_length
            
            current_val = nums[u]
            
            # Check the last time (deepest occurrence) the current node's value appeared on the path from the root.
            # Use .get() with default -1 if the value hasn't been seen before on this path.
            last_occurrence_idx = self.path_values_last_occurrence.get(current_val, -1)
            
            # Determine the start depth for the longest special path ending at the current node u.
            # A special path ending at u cannot include any node at or before the last occurrence of nums[u].
            # Thus, the path must start at the node immediately after the last occurrence.
            start_node_depth = last_occurrence_idx + 1
            
            # Calculate the length and number of nodes for this potential special path ending at u.
            # Path Length = (Length from root to u) - (Length from root to node at start_node_depth).
            candidate_path_len = current_length - self.path_lengths[start_node_depth]
            # Number of Nodes = (Depth of u) - (Depth of start node) + 1.
            candidate_path_nodes = depth - start_node_depth + 1

            # Update the global maximum length found so far and the corresponding minimum number of nodes.
            if candidate_path_len > self.global_max_len:
                # Found a new longest special path. Update max length and reset min nodes count.
                self.global_max_len = candidate_path_len
                self.global_min_nodes = candidate_path_nodes
            elif candidate_path_len == self.global_max_len:
                # Found a path with the same maximum length. Update min nodes if this path has fewer nodes.
                self.global_min_nodes = min(self.global_min_nodes, candidate_path_nodes)

            # Before recursing to children, store the previous state of the last occurrence index for current_val.
            # This is needed for correct backtracking.
            prev_val_idx = self.path_values_last_occurrence.get(current_val, -1)
            
            # Update the map: mark that current_val's last occurrence is now at the current depth.
            self.path_values_last_occurrence[current_val] = depth

            # Recursively call DFS for all children of u.
            for v, length in adj[u]:
                # Ensure traversal moves downwards in the rooted tree structure by not visiting the parent node.
                if v != parent: 
                    dfs(v, u, depth + 1, current_length + length)

            # Backtrack step: Restore the state of path_values_last_occurrence map to what it was before visiting node u.
            # This ensures that when exploring other branches, the map reflects the correct path history.
            if prev_val_idx != -1:
                # If current_val was seen before node u, restore its last occurrence depth to the previous value.
                self.path_values_last_occurrence[current_val] = prev_val_idx
            else:
                # If current_val was first encountered at node u (prev_val_idx == -1),
                # it means this value should be removed from the map upon backtracking out of u's subtree.
                 # Check if the key exists and if its value matches the current depth before deleting.
                 # This guards against potential edge cases, although usually not necessary in standard DFS.
                 if current_val in self.path_values_last_occurrence and self.path_values_last_occurrence[current_val] == depth:
                    del self.path_values_last_occurrence[current_val]

        # Start the DFS traversal from the root node 0.
        # The root has no parent (represented by -1), is at depth 0, and path length from root to itself is 0.
        dfs(0, -1, 0, 0)

        # Return the final computed maximum length and minimum nodes count.
        return [self.global_max_len, self.global_min_nodes]