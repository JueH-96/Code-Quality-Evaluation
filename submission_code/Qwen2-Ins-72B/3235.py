from collections import defaultdict
import heapq

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        graph = defaultdict(lambda: defaultdict(lambda: float('inf')))
        for o, c, co in zip(original, changed, cost):
            graph[o][c] = min(graph[o][c], co)
        
        for k in graph:
            graph[k][k] = 0
        
        for k in graph:
            for i in graph[k]:
                for j in graph[k]:
                    if graph[i][j] > graph[i][k] + graph[k][j]:
                        graph[i][j] = graph[i][k] + graph[k][j]
        
        ans = 0
        for s, t in zip(source, target):
            if graph[s][t] == float('inf'):
                return -1
            ans += graph[s][t]
        
        return ans