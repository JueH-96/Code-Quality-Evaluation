class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        from collections import deque, defaultdict
        
        # Initialize the graph with initial roads
        graph = defaultdict(list)
        for i in range(n - 1):
            graph[i].append(i + 1)
        
        def bfs():
            # BFS to find shortest path from 0 to n-1
            queue = deque([(0, 0)])  # (node, distance)
            visited = set([0])
            
            while queue:
                node, dist = queue.popleft()
                
                if node == n - 1:
                    return dist
                
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            
            return -1  # Should never reach here given the constraints
        
        result = []
        
        for u, v in queries:
            # Add the new road
            graph[u].append(v)
            
            # Find shortest path and add to result
            shortest = bfs()
            result.append(shortest)
        
        return result