from collections import defaultdict, deque
from typing import List

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1  # Number of nodes in the tree
        
        # Build adjacency list representation of the tree
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        result = []
        
        # For each starting node i
        for i in range(n):
            # BFS to find marking times
            marked_time = [-1] * n  # -1 means not marked yet
            marked_time[i] = 0  # Mark the starting node at time 0
            
            queue = deque([i])
            
            while queue:
                node = queue.popleft()
                time = marked_time[node]
                
                for neighbor in graph[node]:
                    if marked_time[neighbor] == -1:  # If not marked yet
                        if neighbor % 2 == 1:  # Odd neighbor
                            marked_time[neighbor] = time + 1
                        else:  # Even neighbor
                            marked_time[neighbor] = time + 2
                        queue.append(neighbor)
            
            # The maximum marking time is when all nodes are marked
            result.append(max(marked_time))
        
        return result