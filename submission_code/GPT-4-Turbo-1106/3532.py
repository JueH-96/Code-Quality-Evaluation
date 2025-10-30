from collections import defaultdict, deque

class Solution:
    def timeTaken(self, edges: list[list[int]]) -> list[int]:
        def bfs(start, is_odd):
            queue = deque([(start, 0)])
            visited = set()
            max_time = 0
            
            while queue:
                node, time = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                max_time = max(max_time, time)
                
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        if is_odd and neighbor % 2 == 0:
                            queue.append((neighbor, time + 2))
                        else:
                            queue.append((neighbor, time + 1))
            
            return max_time
        
        n = len(edges) + 1
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        times = []
        for i in range(n):
            times.append(bfs(i, i % 2 == 1))
        
        return times