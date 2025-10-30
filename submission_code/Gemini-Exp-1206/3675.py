from typing import List
from collections import defaultdict

class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        n = len(edges) + 1
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))

        def dfs(node, parent):
            heap = []
            total_sum = 0
            for neighbor, weight in graph[node]:
                if neighbor != parent:
                    child_sum = dfs(neighbor, node)
                    total_sum += child_sum
                    heap.append(weight)
            
            heap.sort(reverse=True)
            
            
            return total_sum + sum(heap[:k])

        return dfs(0, -1)