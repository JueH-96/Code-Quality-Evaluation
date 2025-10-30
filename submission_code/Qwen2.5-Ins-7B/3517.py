from typing import List

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        from collections import deque
        from heapq import heappop, heappush
        
        # Initialize the graph and the shortest path queue
        graph = {i: [i+1] for i in range(n-1)}
        graph[n-1] = []
        shortest_path = [float('inf')] * n
        shortest_path[0] = 0
        pq = [(0, 0)]
        
        # Function to update the shortest path
        def update_shortest_path(u, v):
            if shortest_path[v] > shortest_path[u] + 1:
                shortest_path[v] = shortest_path[u] + 1
                heappush(pq, (shortest_path[v], v))
        
        # Process each query
        result = []
        for u, v in queries:
            # Add the new road to the graph
            graph[u].append(v)
            graph[v].append(u)
            
            # Update the shortest path for the new road
            update_shortest_path(u, v)
            update_shortest_path(v, u)
            
            # Process the shortest path queue
            while pq:
                dist, node = heappop(pq)
                if dist > shortest_path[node]:
                    continue
                for neighbor in graph[node]:
                    update_shortest_path(node, neighbor)
            
            # Append the shortest path from 0 to n-1 to the result
            result.append(shortest_path[n-1])
        
        return result