class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        from collections import deque, defaultdict
        
        # Initialize the graph with initial roads i -> i+1
        graph = defaultdict(list)
        for i in range(n - 1):
            graph[i].append(i + 1)
        
        def bfs():
            """Find shortest path from 0 to n-1 using BFS"""
            queue = deque([0])
            visited = set([0])
            distance = 0
            
            while queue:
                size = len(queue)
                for _ in range(size):
                    node = queue.popleft()
                    if node == n - 1:
                        return distance
                    
                    for neighbor in graph[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                distance += 1
            
            return -1  # Should never reach here
        
        result = []
        for u, v in queries:
            graph[u].append(v)
            result.append(bfs())
        
        return result