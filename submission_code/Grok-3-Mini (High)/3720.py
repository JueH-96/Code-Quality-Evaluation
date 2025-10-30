import collections
from typing import List

class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
        def can_reach_all(max_weight: int) -> bool:
            # Build adjacency list for the reversed graph
            adj_reversed = [[] for _ in range(n)]
            for a, b, w in edges:
                if w <= max_weight:
                    # Original edge a -> b, reversed edge b -> a
                    adj_reversed[b].append(a)  # In reversed graph, from b to a
            
            # BFS from node 0 in the reversed graph to check reachability
            visited = [False] * n
            queue = collections.deque()
            queue.append(0)
            visited[0] = True
            nodes_visited = 1  # Start with node 0 visited
            
            while queue:
                node = queue.popleft()
                for neighbor in adj_reversed[node]:  # Neighbors in reversed graph
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
                        nodes_visited += 1
            
            return nodes_visited == n  # All nodes should be reachable from 0 in reversed graph
        
        # Binary search on the maximum edge weight
        left, right = 1, 1000000  # Min and max possible weights
        result = -1
        while left <= right:
            mid = (left + right) // 2
            if can_reach_all(mid):
                result = mid
                right = mid - 1  # Try to minimize the weight
            else:
                left = mid + 1  # Need a larger weight
        
        return result