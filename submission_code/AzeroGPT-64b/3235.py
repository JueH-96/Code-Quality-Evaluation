from collections import defaultdict
from heapq import heappush, heappop

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        n = len(source)
        graph = defaultdict(dict)
        for orig, chng, c in zip(original, changed, cost):
            if orig not in graph[chng] or graph[chng][orig] > c:
                graph[orig][chng] = c
                graph[chng][orig] = c
        
        def dijkstra(s, d):
            if s not in graph and d not in graph:
                return -inf
            
            dist = defaultdict(lambda: inf)
            dist[s] = 0
            heap = [(0, s)]
            
            while heap:
                cd, ch = heappop(heap)
                if cd > dist[ch]:
                    continue
                
                for nh, nd in graph[ch].items():
                    if dist[nh] > dist[ch] + nd:
                        dist[nh] = dist[ch] + nd
                        heappush(heap, (dist[nh], nh))
            
            return dist[d]
                
        total_cost = 0
        for s, t in zip(source, target):
            cost_to_target = 0 if s == t else dijkstra(s, t)
            if cost_to_target == inf:
                return -1
            total_cost += cost_to_target
        
        return total_cost