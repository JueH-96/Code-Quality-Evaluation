import collections
import sys

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1  # Number of nodes (0 to n-1)
        
        # Increase recursion limit to handle deep trees (up to 10^5 nodes)
        sys.setrecursionlimit(n + 100) 

        # Build adjacency list for the tree
        adj = collections.defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Define the cost for a node to be marked
        # node_costs[i] = 1 if i is odd, 2 if i is even
        # This is the cost to "arrive" at node i from any adjacent marked node.
        node_costs = [1 if i % 2 != 0 else 2 for i in range(n)]

        # dfs_dist[u]: Stores the maximum time a node in the subtree rooted at 'u'
        # will be marked, assuming 'u' is the starting node (t=0) and propagation
        # only happens downwards into 'u's subtree. The sum includes the cost of
        # the furthest node 'v' but excludes the cost of 'u'.
        dfs_dist = [0] * n

        # up_max_dist[u]: Stores the maximum time a node outside the subtree rooted at 'u'
        # will be marked, assuming 'u' is the starting node (t=0) and propagation
        # happens upwards from 'u' towards its parent and then potentially to other subtrees.
        # The sum includes the cost of the furthest node 'v' but excludes the cost of 'u'.
        up_max_dist = [0] * n
        
        # ans[i]: Stores the final result for each node i being the starting node.
        # This is the maximum of dfs_dist[i] and up_max_dist[i].
        ans = [0] * n

        # DFS1: Computes dfs_dist for all nodes using a post-order traversal
        def dfs1(u: int, parent: int):
            current_max_down_path = 0
            for v in adj[u]:
                if v != parent:
                    dfs1(v, u)
                    # The path to a child 'v' (dfs_dist[v] + node_costs[v]) contributes to u's max downward path
                    current_max_down_path = max(current_max_down_path, dfs_dist[v] + node_costs[v])
            dfs_dist[u] = current_max_down_path

        # DFS2: Computes up_max_dist for all nodes and the final answer using a pre-order traversal
        def dfs2(u: int, parent: int):
            # For each child 'v' of 'u', up_max_dist[v] depends on the max path
            # that goes up from 'u' (up_max_dist[u]) OR down into a sibling's subtree.
            # We need to find the top two longest paths going downwards from 'u' to its children.

            child_path_values = [] # List of (path_length, child_node_id)
            for v in adj[u]:
                if v != parent:
                    child_path_values.append((dfs_dist[v] + node_costs[v], v))
            
            # Sort children by their path values in descending order
            child_path_values.sort(key=lambda x: x[0], reverse=True)

            # Get the longest and second longest downward paths from 'u'
            max1_val = 0
            max2_val = 0
            if len(child_path_values) >= 1:
                max1_val = child_path_values[0][0]
            if len(child_path_values) >= 2:
                max2_val = child_path_values[1][0]

            for v in adj[u]:
                if v != parent:
                    # Calculate max_from_u_elsewhere: the max path starting at 'u'
                    # (excluding 'u's cost), that does NOT go into 'v's subtree.
                    # This path can go up to 'u's parent or down to 'u's other children (siblings of 'v').
                    max_from_u_elsewhere = up_max_dist[u] # Path going up from u

                    # If v's path was the longest downward path from u, use the second longest.
                    # Otherwise, use the longest.
                    if child_path_values and child_path_values[0][1] == v:
                        max_from_u_elsewhere = max(max_from_u_elsewhere, max2_val)
                    else:
                        max_from_u_elsewhere = max(max_from_u_elsewhere, max1_val)
                    
                    # up_max_dist[v] is the sum of:
                    # 1. The cost to reach 'u' from 'v' (node_costs[u])
                    # 2. The maximum path from 'u' that does not involve 'v's subtree.
                    up_max_dist[v] = node_costs[u] + max_from_u_elsewhere
                    
                    # Recursively call dfs2 for child 'v'
                    dfs2(v, u)
            
            # The total max time for node 'u' to be the start is the max of:
            # - The furthest node in its own subtree (dfs_dist[u])
            # - The furthest node outside its subtree (up_max_dist[u])
            ans[u] = max(dfs_dist[u], up_max_dist[u])

        # Start the DFS traversals from an arbitrary root (e.g., node 0)
        # Assuming the graph is connected (a valid tree), all nodes will be visited.
        dfs1(0, -1)  # -1 represents a non-existent parent for the root
        dfs2(0, -1)

        return ans