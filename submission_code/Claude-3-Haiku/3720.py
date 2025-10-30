from collections import defaultdict
import heapq

class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
        # Build the graph
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[u].append((v, w))
        
        # Dijkstra's algorithm to find the minimum max weight
        def dijkstra(start):
            pq = [(0, start)]
            dist = [float('inf')] * n
            dist[start] = 0
            
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                
                for v, w in graph[u]:
                    new_dist = max(dist[u], w)
                    if new_dist < dist[v]:
                        dist[v] = new_dist
                        heapq.heappush(pq, (new_dist, v))
            
            return dist
        
        # Find the minimum max weight
        min_max_weight = float('inf')
        for i in range(n):
            if len(graph[i]) <= threshold:
                dist = dijkstra(i)
                if all(d < float('inf') for d in dist):
                    min_max_weight = min(min_max_weight, max(dist))
        
        return min_max_weight if min_max_weight != float('inf') else -1