from typing import List
from collections import deque

class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
        original_edges_sorted = [[] for _ in range(n)]
        max_weight = 0
        for a, b, w in edges:
            original_edges_sorted[a].append((w, b))
            if w > max_weight:
                max_weight = w
        # Sort each node's edges by weight
        for u in range(n):
            original_edges_sorted[u].sort()
        
        low = 1
        high = max_weight
        answer = -1
        
        while low <= high:
            mid = (low + high) // 2
            if self.is_feasible(mid, n, original_edges_sorted, edges, threshold):
                answer = mid
                high = mid - 1
            else:
                low = mid + 1
        return answer
    
    def is_feasible(self, W: int, n: int, original_edges_sorted: List[List[tuple]], edges: List[List[int]], threshold: int) -> bool:
        reversed_adj = [[] for _ in range(n)]
        for a, b, w in edges:
            if w <= W:
                reversed_adj[b].append(a)
        
        # BFS to compute distances from 0 in the reversed graph
        INF = float('inf')
        distance = [INF] * n
        q = deque()
        distance[0] = 0
        q.append(0)
        
        while q:
            u = q.popleft()
            for v in reversed_adj[u]:
                if distance[v] == INF:
                    distance[v] = distance[u] + 1
                    q.append(v)
        
        # Check if any node is unreachable
        for d in distance:
            if d == INF:
                return False
        
        # Check each node except 0
        for u in range(n):
            if u == 0:
                continue
            
            edges_list = original_edges_sorted[u]
            left, right = 0, len(edges_list)
            # Binary search to find all edges with weight <= W
            while left < right:
                mid = (left + right) // 2
                if edges_list[mid][0] <= W:
                    left = mid + 1
                else:
                    right = mid
            edges_u = [(b, w) for (w, b) in edges_list[:left]]
            
            # Sort edges by the distance of their destination
            edges_u.sort(key=lambda x: distance[x[0]])
            selected_edges = edges_u[:threshold]
            
            has_path = False
            for (v, w) in selected_edges:
                if distance[v] < distance[u]:
                    has_path = True
                    break
            if not has_path:
                return False
        
        return True