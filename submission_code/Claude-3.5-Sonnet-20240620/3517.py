class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize the graph with initial roads
        graph = {i: [i+1] for i in range(n-1)}
        graph[n-1] = []

        # Function to find the shortest path using BFS
        def bfs():
            queue = [(0, 0)]
            visited = set()
            while queue:
                node, dist = queue.pop(0)
                if node == n - 1:
                    return dist
                if node in visited:
                    continue
                visited.add(node)
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        queue.append((neighbor, dist + 1))
            return -1  # If no path found

        answer = []
        for u, v in queries:
            # Add the new road
            if u not in graph:
                graph[u] = []
            graph[u].append(v)
            
            # Find the shortest path after adding the new road
            shortest_path = bfs()
            answer.append(shortest_path)

        return answer