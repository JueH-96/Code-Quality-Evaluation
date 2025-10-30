from collections import defaultdict
import heapq

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        n = len(source)
        graph = defaultdict(dict)
        
        for o, c, co in zip(original, changed, cost):
            if c not in graph[o] or co < graph[o][c]:
                graph[o][c] = co
        
        def dijkstra(start, end):
            dist = {char: float('inf') for char in graph}
            dist[start] = 0
            pq = [(0, start)]
            
            while pq:
                current_dist, current_char = heapq.heappop(pq)
                
                if current_dist > dist[current_char]:
                    continue
                
                for neighbor, weight in graph[current_char].items():
                    distance = current_dist + weight
                    
                    if distance < dist[neighbor]:
                        dist[neighbor] = distance
                        heapq.heappush(pq, (distance, neighbor))
            
            return dist[end] if dist[end] != float('inf') else -1
        
        total_cost = 0
        for i in range(n):
            if source[i] != target[i]:
                min_cost = dijkstra(source[i], target[i])
                if min_cost == -1:
                    return -1
                total_cost += min_cost
        
        return total_cost