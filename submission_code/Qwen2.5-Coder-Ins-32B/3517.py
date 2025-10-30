from typing import List
import heapq

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        def dijkstra(graph, start, end):
            distances = [float('inf')] * n
            distances[start] = 0
            priority_queue = [(0, start)]
            
            while priority_queue:
                current_distance, current_node = heapq.heappop(priority_queue)
                
                if current_distance > distances[current_node]:
                    continue
                
                for neighbor, weight in graph[current_node]:
                    distance = current_distance + weight
                    
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        heapq.heappush(priority_queue, (distance, neighbor))
            
            return distances[end]
        
        graph = {i: [] for i in range(n)}
        for i in range(n - 1):
            graph[i].append((i + 1, 1))
        
        result = []
        for u, v in queries:
            graph[u].append((v, 1))
            result.append(dijkstra(graph, 0, n - 1))
        
        return result