from collections import defaultdict, deque

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        res = []
        for start in range(n):
            visited = [False] * n
            queue = deque([(start, 0)])
            visited[start] = True
            max_time = 0
            
            while queue:
                node, time = queue.popleft()
                if node % 2 == 0 and time >= 2 and any(visited[neighbor] for neighbor in graph[node]):
                    max_time = max(max_time, time)
                elif node % 2 == 1 and time >= 1 and any(visited[neighbor] for neighbor in graph[node]):
                    max_time = max(max_time, time)
                
                for neighbor in graph[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append((neighbor, time + 1))
            
            res.append(max_time)
        
        return res