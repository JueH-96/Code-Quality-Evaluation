from collections import defaultdict, deque
from typing import List

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        results = []
        for start_node in range(n):
            times = [-1] * n
            times[start_node] = 0
            q = deque([(start_node, 0)])
            max_time = 0
            
            while q:
                curr_node, curr_time = q.popleft()
                max_time = max(max_time, curr_time)
                
                for neighbor in adj[curr_node]:
                    if times[neighbor] == -1:
                        if neighbor % 2 == 1:
                            times[neighbor] = curr_time + 1
                            q.append((neighbor, curr_time + 1))
                        else:
                            times[neighbor] = curr_time + 2
                            q.append((neighbor, curr_time + 2))
            results.append(max_time)
        return results