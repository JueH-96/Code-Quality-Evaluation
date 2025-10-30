from typing import List
import collections

class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
        
        def check(max_w: int) -> bool:
            """
            Checks if it's possible to satisfy the conditions with a maximum edge weight of max_w.
            This function determines if a valid subgraph can be constructed using only edges with weight <= max_w.
            
            The condition "all nodes can reach node 0" is equivalent to "in the reversed graph,
            node 0 can reach all other nodes". We can check this with a traversal (like BFS) starting from node 0,
            while respecting the out-degree threshold.
            """
            # Since edge weights are at least 1, a max_w < 1 means no edges can be used.
            # With n >= 2, it's impossible for other nodes to reach node 0.
            if max_w < 1:
                return False

            # Build a reversed graph considering only edges with weight <= max_w.
            # adj_rev[v] will contain a list of nodes 'u' for each original edge u -> v.
            adj_rev = [[] for _ in range(n)]
            for u, v, w in edges:
                if w <= max_w:
                    adj_rev[v].append(u)

            # 'q' will store nodes that are confirmed to be able to reach node 0.
            q = collections.deque([0])
            
            # 'reachable' stores the set of nodes for which a path to 0 has been found.
            reachable = {0}
            
            # 'out_degree_used[u]' tracks how many outgoing edges from node 'u' we have selected.
            out_degree_used = [0] * n

            # The algorithm works by expanding the set of reachable nodes, starting from node 0.
            # A node 'v' in the queue means we have a path for it to 0. We can then use 'v'
            # to connect other nodes that have an edge to 'v'.
            while q:
                v = q.popleft()

                # For each node 'u' that has an edge to 'v' (u -> v in original graph)...
                for u in adj_rev[v]:
                    # If 'u' is not yet known to be reachable...
                    if u not in reachable:
                        # We can establish a path for 'u' via 'v' if 'u' has not
                        # exceeded its out-degree threshold.
                        if out_degree_used[u] < threshold:
                            out_degree_used[u] += 1
                            reachable.add(u)
                            q.append(u)
            
            # If all n nodes are in the 'reachable' set, it is possible.
            return len(reachable) == n

        # Binary search for the minimum possible value of the maximum edge weight.
        # The lowest possible weight is 1, max is 10^6.
        # `low` is inclusive, `high` is exclusive.
        low = 0
        high = 10**6 + 2  # A safe upper bound beyond the max possible weight
        ans = -1

        while low < high:
            mid = low + (high - low) // 2
            
            if check(mid):
                # 'mid' is a potential answer. We try for an even smaller weight.
                ans = mid
                high = mid
            else:
                # 'mid' is too small, we must allow larger edge weights.
                low = mid + 1
                
        return ans