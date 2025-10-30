from typing import List
from collections import deque

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize the graph with the initial roads
        graph = {i: [i + 1] for i in range(n - 1)}
        graph[n - 1] = []  # No outgoing roads from the last city

        def bfs_shortest_path():
            # Perform BFS to find the shortest path from city 0 to city n-1
            queue = deque([(0, 0)])  # (current_city, current_distance)
            visited = set()
            while queue:
                current_city, current_distance = queue.popleft()
                if current_city == n - 1:
                    return current_distance
                if current_city in visited:
                    continue
                visited.add(current_city)
                for neighbor in graph[current_city]:
                    if neighbor not in visited:
                        queue.append((neighbor, current_distance + 1))
            return float('inf')  # If no path is found, though it should always be found

        answer = []
        for u, v in queries:
            # Add the new road to the graph
            if u in graph:
                graph[u].append(v)
            else:
                graph[u] = [v]
            # Calculate the shortest path after adding the new road
            shortest_path_length = bfs_shortest_path()
            answer.append(shortest_path_length)

        return answer