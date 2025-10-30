from collections import deque

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize the graph with the initial roads
        graph = {i: [i+1] for i in range(n-1)}
        graph[n-1] = []
        
        # Initialize the answer list
        answer = []
        
        # Precompute the initial shortest path
        # Since the initial roads form a straight line, the shortest path is n-1 steps
        shortest = n-1
        
        for u, v in queries:
            # Add the new road to the graph
            if u in graph:
                graph[u].append(v)
            else:
                graph[u] = [v]
            
            # Perform BFS to find the shortest path from 0 to n-1
            visited = {0: 0}
            queue = deque([0])
            found = False
            while queue:
                current = queue.popleft()
                if current == n-1:
                    found = True
                    break
                for neighbor in graph.get(current, []):
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            
            if found:
                shortest = visited[n-1]
            else:
                # If no path found, it means the initial path is still the shortest
                shortest = n-1
            
            answer.append(shortest)
        
        return answer