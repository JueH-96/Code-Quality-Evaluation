from typing import List
from collections import deque, defaultdict

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        graph = defaultdict(list)
        
        # Build the graph from the edges
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def bfs(start: int) -> int:
            # Initialize the marking times with -1 (unmarked)
            mark_time = [-1] * n
            mark_time[start] = 0
            queue = deque([start])
            
            while queue:
                node = queue.popleft()
                current_time = mark_time[node]
                
                for neighbor in graph[node]:
                    if mark_time[neighbor] == -1:
                        # Determine the time when the neighbor will be marked
                        if neighbor % 2 == 0:
                            # Even node
                            mark_time[neighbor] = current_time + 2
                        else:
                            # Odd node
                            mark_time[neighbor] = current_time + 1
                        queue.append(neighbor)
            
            # The time when the last node gets marked
            return max(mark_time)
        
        # Calculate the time taken for each node to mark all others
        result = []
        for i in range(n):
            result.append(bfs(i))
        
        return result