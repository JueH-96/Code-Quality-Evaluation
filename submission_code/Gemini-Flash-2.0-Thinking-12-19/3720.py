import collections
from typing import List

class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:

        # Check function: Is it possible to satisfy conditions using only edges with weight <= max_w?
        def check(max_w: int) -> bool:
            # Build the reversed graph with edges having weight <= max_w
            # Edge u -> v with weight w in original becomes v -> u with weight w in reversed
            adj_rev = [[] for _ in range(n)]
            for u, v, w in edges:
                if w <= max_w:
                    adj_rev[v].append(u) # Reversed edge v -> u

            # BFS to find reachability from node 0 in the reversed graph
            # subject to in-degree constraint (which is original out-degree constraint)
            
            # `remaining_in_degree[u]` stores how many more incoming edges node `u` can accept
            # in the selected subgraph in the reversed graph. This corresponds to the
            # remaining out-degree capacity for node `u` in the original graph.
            remaining_in_degree = [threshold] * n
            
            # `reachable[u]` is True if node `u` is reachable from 0 in the reversed graph
            # using the selected edges.
            reachable = [False] * n
            reachable[0] = True # Node 0 is the starting point in the reversed graph
            
            # Queue for BFS, stores nodes reachable from 0 in the reversed graph
            q = collections.deque([0])
            
            # visited_count tracks how many nodes are reachable from 0
            visited_count = 1

            while q:
                v = q.popleft()

                # Optimization: If all nodes are visited, we can stop early
                if visited_count == n:
                    return True

                # In the reversed graph, we look at neighbors of v (nodes u such that v -> u exists)
                # These correspond to original edges u -> v
                for u in adj_rev[v]:
                    
                    # Can we use the edge v -> u (in G_rev) ?
                    # This edge contributes to the in-degree of u in G_rev.
                    # The constraint is that u's in-degree in G_rev must be <= threshold.
                    
                    # If node u can accept more incoming edges
                    if remaining_in_degree[u] > 0:
                        remaining_in_degree[u] -= 1 # Use one incoming slot for u
                        
                        # If u was not reachable before, it is now
                        if not reachable[u]:
                            reachable[u] = True
                            visited_count += 1
                            q.append(u) # Add u to the queue to explore further
                            
                            # Optimization: If all nodes become visited, return True immediately
                            if visited_count == n:
                                return True

            # After BFS, check if all nodes are reachable from node 0 in the reversed graph
            return visited_count == n

        # Binary search for the minimum maximum edge weight
        # The possible weights are from 1 up to the maximum weight in edges (or 10^6).
        # We search for the minimum `max_w` such that `check(max_w)` is True.
        # The range of possible answers is [1, 10^6]. If no such max_w exists, answer is -1.
        # We use a binary search range [1, 10^6 + 1) to handle the case where it's impossible.
        # If check(10^6) is False, the answer is impossible (-1).
        # If check(1) is True, the answer is 1.
        low = 1
        high = 10**6 + 1 # Exclusive upper bound. Represents impossible if low reaches here.
        ans = -1 # Initialize answer to -1

        # The binary search finds the smallest `mid` in the range [low, high)
        # for which check(mid) is True.
        while low < high:
            mid = low + (high - low) // 2

            if check(mid):
                # If possible with max_w = mid, it means mid is a potential answer.
                # We try a smaller maximum weight.
                ans = mid # mid is a candidate answer
                high = mid # Try smaller weights in [low, mid)
            else:
                # If not possible with max_w = mid, we need to allow larger weights.
                # The answer must be > mid.
                low = mid + 1 # Search in [mid + 1, high)
                
        # After the loop, low == high.
        # If a valid max_w was found, ans will be the minimum such value.
        # If no valid max_w was found (check(10^6) was False), low will become 10^6 + 1,
        # and ans will remain -1.
        return ans