from collections import deque

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize the graph with the initial roads
        graph = {i: [i+1] for i in range(n-1)}
        graph[n-1] = []
        
        # Initialize the answer list
        answer = []
        
        for u, v in queries:
            # Add the new road to the graph
            if u in graph:
                graph[u].append(v)
            else:
                graph[u] = [v]
            
            # Perform BFS to find the shortest path from 0 to n-1
            visited = {0: 0}
            queue = deque([0])
            while queue:
                current = queue.popleft()
                if current == n-1:
                    break
                for neighbor in graph.get(current, []):
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            
            # Append the shortest path length to the answer
            if n-1 in visited:
                answer.append(visited[n-1])
            else:
                answer.append(-1)  # If no path exists, though constraints ensure it does
        
        return answer