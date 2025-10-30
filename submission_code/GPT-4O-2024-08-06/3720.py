from typing import List
from collections import defaultdict, deque

class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
        def canFormGraphWithMaxWeight(maxWeight):
            # Create adjacency list with edges having weight <= maxWeight
            adj = defaultdict(list)
            for u, v, w in edges:
                if w <= maxWeight:
                    adj[u].append((v, w))
            
            # Check if node 0 is reachable from all other nodes
            visited = [False] * n
            queue = deque([0])
            visited[0] = True
            while queue:
                node = queue.popleft()
                for neighbor, weight in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            
            # If any node is not visited, return False
            if not all(visited):
                return False
            
            # Check if each node has at most 'threshold' outgoing edges
            for node in range(n):
                if len(adj[node]) > threshold:
                    return False
            
            return True
        
        # Binary search over the possible maximum edge weight
        left, right = 1, max(w for _, _, w in edges)
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            if canFormGraphWithMaxWeight(mid):
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return result