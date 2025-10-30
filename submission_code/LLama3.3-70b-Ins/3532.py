from collections import defaultdict, deque
from typing import List

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        graph = defaultdict(list)
        
        # Build the graph
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        times = []
        
        # For each node, calculate the time taken to mark all nodes
        for i in range(n):
            time = [0] * n
            queue = deque([(i, 0)])
            visited = {i}
            
            while queue:
                node, t = queue.popleft()
                time[node] = t
                
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        if node % 2 == 0:
                            if time[neighbor] == 0 or time[neighbor] > t + 2:
                                queue.append((neighbor, t + 2))
                                visited.add(neighbor)
                        else:
                            if time[neighbor] == 0 or time[neighbor] > t + 1:
                                queue.append((neighbor, t + 1))
                                visited.add(neighbor)
            
            times.append(max(time))
        
        return times