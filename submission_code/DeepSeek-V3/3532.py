from collections import deque

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        times = [0] * n
        for i in range(n):
            time = [0] * n
            q = deque()
            q.append(i)
            time[i] = 0
            while q:
                node = q.popleft()
                current_time = time[node]
                for neighbor in adj[node]:
                    if time[neighbor] == 0 and neighbor != i:
                        if neighbor % 2 == 1:
                            time[neighbor] = current_time + 1
                        else:
                            time[neighbor] = current_time + 2
                        q.append(neighbor)
            times[i] = max(time)
        return times