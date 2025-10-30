from typing import List
import heapq

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize the graph with the initial roads
        graph = {i: [i + 1] for i in range(n - 1)}
        graph[n - 1] = []
        
        def dijkstra():
            # Dijkstra's algorithm to find the shortest path from 0 to n-1
            distances = [float('inf')] * n
            distances[0] = 0
            pq = [(0, 0)]  # (distance, node)
            
            while pq:
                current_distance, current_node = heapq.heappop(pq)
                
                if current_distance > distances[current_node]:
                    continue
                
                for neighbor in graph[current_node]:
                    distance = current_distance + 1
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        heapq.heappush(pq, (distance, neighbor))
            
            return distances[n - 1]
        
        answer = []
        for u, v in queries:
            if u in graph:
                graph[u].append(v)
            else:
                graph[u] = [v]
            shortest_path_length = dijkstra()
            answer.append(shortest_path_length)
        
        return answer