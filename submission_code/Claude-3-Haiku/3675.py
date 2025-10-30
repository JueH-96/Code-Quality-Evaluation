from collections import defaultdict
from heapq import heappush, heappop

class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        n = len(edges) + 1
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))
        
        def dfs(node, parent):
            heap = []
            for neighbor, weight in graph[node]:
                if neighbor != parent:
                    heappush(heap, (-weight, neighbor))
            
            total_weight = 0
            while len(heap) > k:
                _, neighbor = heappop(heap)
                total_weight += -heappop(heap)[0]
                dfs(neighbor, node)
            
            for _, neighbor in heap:
                dfs(neighbor, node)
            
            return total_weight
        
        return dfs(0, -1)