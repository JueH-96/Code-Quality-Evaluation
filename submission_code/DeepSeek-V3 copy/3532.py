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
            visited = [False] * n
            time = [0] * n
            q = deque()
            q.append(i)
            visited[i] = True
            time[i] = 0
            while q:
                node = q.popleft()
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        if neighbor % 2 == 1:
                            if time[node] + 1 >= time[neighbor]:
                                time[neighbor] = time[node] + 1
                        else:
                            if time[node] + 2 >= time[neighbor]:
                                time[neighbor] = time[node] + 2
                        visited[neighbor] = True
                        q.append(neighbor)
            times[i] = max(time)
        return times