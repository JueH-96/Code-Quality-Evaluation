from typing import List
from collections import defaultdict, deque

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize the graph with the initial roads
        graph = defaultdict(list)
        for i in range(n - 1):
            graph[i].append(i + 1)

        # Function to perform BFS and find the shortest path from city 0 to city n-1
        def bfs():
            queue = deque([0])
            visited = set([0])
            distance = 0
            while queue:
                for _ in range(len(queue)):
                    node = queue.popleft()
                    if node == n - 1:
                        return distance
                    for neighbor in graph[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                distance += 1
            return float('inf')  # Return infinity if no path is found

        # Initialize the result list
        result = []

        # Process each query
        for u, v in queries:
            graph[u].append(v)
            result.append(bfs())

        return result