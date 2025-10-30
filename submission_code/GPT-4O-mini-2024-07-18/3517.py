from typing import List
import heapq

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize the graph with the default roads
        graph = {i: [] for i in range(n)}
        for i in range(n - 1):
            graph[i].append(i + 1)
        
        # Function to calculate the shortest path from 0 to n-1 using Dijkstra's algorithm
        def dijkstra():
            distances = [float('inf')] * n
            distances[0] = 0
            min_heap = [(0, 0)]  # (distance, node)
            
            while min_heap:
                current_distance, current_node = heapq.heappop(min_heap)
                
                if current_distance > distances[current_node]:
                    continue
                
                for neighbor in graph[current_node]:
                    distance = current_distance + 1
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        heapq.heappush(min_heap, (distance, neighbor))
            
            return distances[n - 1]
        
        answer = []
        for u, v in queries:
            graph[u].append(v)  # Add the new road
            shortest_path_length = dijkstra()  # Calculate the shortest path
            answer.append(shortest_path_length)
        
        return answer