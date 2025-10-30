from typing import List
from collections import deque

class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:

        # Function to check if it's possible to satisfy conditions
        # with maximum edge weight <= max_w
        def check(max_w: int) -> bool:
            # Build the transpose graph adj_T_out: adj_T_out[u] contains v if original edge v -> u exists
            # with weight <= max_w.
            # This represents edges u <- v in the original graph, which are edges u -> v in the transpose graph.
            # We need reachability u -> ... -> 0 in Original (G) for all u != 0.
            # This is reachability 0 -> ... -> u in Transpose (G_T) for all u != 0.
            # Constraint: out-degree_G(u) <= threshold for all u.
            # This is in-degree_G_T(u) <= threshold for all u.

            adj_T_out = {i: [] for i in range(n)}
            for u_orig, v_orig, w in edges:
                if w <= max_w:
                    # Original u_orig -> v_orig. Transpose v_orig -> u_orig.
                    # adj_T_out[source_T].append(dest_T)
                    adj_T_out[v_orig].append(u_orig)

            # Use BFS starting from node 0 in the transpose graph adj_T_out.
            # We find the set of nodes reachable from 0 in G_T_s (subgraph of G_T)
            # respecting the in-degree constraint in G_T_s.
            can_be_reached = {0}
            # in_degree_count[v] stores the number of selected incoming edges for v in G_T_s.
            # This corresponds to the number of selected outgoing edges from v in G_s.
            # This count must be <= threshold for all v.
            in_degree_count = [0] * n # Use list for O(1) access
            q = deque([0])

            while q:
                u = q.popleft() # u is a node reached from 0 in G_T_s

                # Neighbors of u in adj_T_out are nodes v such that transpose edge u -> v exists.
                # These are the nodes v such that original edge v -> u exists with weight <= max_w.
                neighbors_in_T = adj_T_out.get(u, [])
                for v in neighbors_in_T:
                    # u -> v is a transpose edge.
                    # If v is not yet reached from 0 in G_T_s, and v can accept one more incoming edge in G_T_s
                    # (i.e., v has remaining out-degree capacity in G_s).
                    if v not in can_be_reached and in_degree_count[v] < threshold:
                        can_be_reached.add(v)
                        in_degree_count[v] += 1 # We select one incoming edge for v in G_T_s
                        q.append(v)

            # Check if all nodes 0 to n-1 are reachable from 0 in G_T_s.
            # If so, all nodes u != 0 can reach 0 in G_s with out-degree constraints satisfied.
            return len(can_be_reached) == n

        # Binary search for the minimum max_w
        # Possible edge weights are 1 to 10^6.
        low = 1
        high = 10**6 + 1 # Search space is [1, 10^6] inclusive. high is exclusive.
        ans = -1 # Default return value if impossible

        while low < high:
            mid = low + (high - low) // 2
            if check(mid):
                ans = mid # mid is a possible maximum weight, try smaller
                high = mid
            else:
                low = mid + 1 # mid is not possible, need larger maximum weight

        return ans