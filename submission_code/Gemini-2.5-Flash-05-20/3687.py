import collections
from typing import List

class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        n = len(nums)
        adj = collections.defaultdict(list)
        for u, v, length in edges:
            adj[u].append((v, length))
            adj[v].append((u, length))

        # Global variables for results
        # max_overall_len: stores the maximum length found so far
        # min_overall_nodes: stores the minimum number of nodes for max_overall_len
        # Initialized for a single-node path (length 0, 1 node)
        self.max_overall_len = 0
        self.min_overall_nodes = 1

        # Helper function to update the global max length and min nodes count
        def update_global_max(current_len, current_nodes):
            if current_len > self.max_overall_len:
                self.max_overall_len = current_len
                self.min_overall_nodes = current_nodes
            elif current_len == self.max_overall_len:
                self.min_overall_nodes = min(self.min_overall_nodes, current_nodes)

        # dfs(u, p) performs a Depth-First Search starting from node u,
        # with p as its parent.
        # It returns a dictionary `paths_from_u` where:
        #   - Keys are `nums[w]` (the value of an ending node `w` in a special path).
        #   - Values are `(length, num_nodes)` for the longest special path from `u` to `w`.
        def dfs(u, p):
            # paths_from_u: Stores special paths starting at `u`.
            # Key: nums[ending_node_of_path]
            # Value: (length_from_u_to_ending_node, num_nodes_from_u_to_ending_node)
            
            # Initialize with the path consisting only of node `u` itself.
            # This path has length 0 and 1 node.
            paths_from_u = {nums[u]: (0, 1)}
            
            # Update global max results for the path `u`
            update_global_max(0, 1) 

            # Iterate through neighbors of `u`
            for v, length in adj[u]:
                if v == p:
                    # Skip the parent node to ensure downward traversal in the rooted tree
                    continue

                # Recursively call DFS for child `v` to get paths starting from `v`
                child_paths = dfs(v, u)

                # Combine paths from `u` with paths from `v`'s subtree
                # For each path `v -> ... -> w` (represented by `val_w` in `child_paths`):
                # We attempt to form a new path `u -> v -> ... -> w`.
                for val_w, (len_v_w, nodes_v_w) in child_paths.items():
                    # Check the uniqueness constraint: `nums[u]` must not be present
                    # in the path `v -> ... -> w`. `val_w` is `nums[w]`, so if `val_w == nums[u]`,
                    # it means `nums[u]` is already present in the path `v -> ... -> w` (at node `w`),
                    # or it implies `nums[u] == nums[v]` if `val_w` is `nums[v]` for path `v`.
                    # Crucially, `paths_from_v` guarantees uniqueness within `v -> ... -> w`.
                    # We just need to check `nums[u]` against all values in `v -> ... -> w`.
                    # The check `val_w == nums[u]` effectively ensures this, as `val_w` represents
                    # a value in the path `v -> ... -> w`. If `nums[v]` itself is `nums[u]`, then
                    # the path `v` in `child_paths` would have `val_w = nums[v] = nums[u]`.
                    if val_w == nums[u]:
                        continue # This path cannot be extended from `u` due to duplicate value

                    # Calculate length and node count for the new path `u -> v -> ... -> w`
                    current_path_len = length + len_v_w # `length` is the edge length between u and v
                    current_path_nodes = 1 + nodes_v_w

                    # Update the global maximum length and minimum node count
                    update_global_max(current_path_len, current_path_nodes)

                    # Update `paths_from_u`:
                    # We want to store the longest path from `u` that ends at a node with value `val_w`.
                    # If multiple paths from `u` end at nodes with the same value `val_w`,
                    # we pick the one with the maximum length. If lengths are tied, pick minimum nodes.
                    if val_w not in paths_from_u or current_path_len > paths_from_u[val_w][0]:
                        paths_from_u[val_w] = (current_path_len, current_path_nodes)
                    elif current_path_len == paths_from_u[val_w][0]:
                        paths_from_u[val_w] = (current_path_len, min(paths_from_u[val_w][1], current_path_nodes))
            
            # Return the collected paths that start at `u`
            return paths_from_u

        # Start the DFS from node 0 (the root), with no parent (-1)
        dfs(0, -1)
        
        # Return the final results
        return [self.max_overall_len, self.min_overall_nodes]