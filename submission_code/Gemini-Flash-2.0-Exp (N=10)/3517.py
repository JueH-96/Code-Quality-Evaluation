from collections import deque

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        adj = [[] for _ in range(n)]
        for i in range(n - 1):
            adj[i].append(i + 1)
        
        ans = []
        for query in queries:
            u, v = query
            adj[u].append(v)
            
            q = deque([(0, 0)])
            visited = {0}
            min_dist = float('inf')
            
            while q:
                curr, dist = q.popleft()
                if curr == n - 1:
                    min_dist = dist
                    break
                
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, dist + 1))
            
            ans.append(min_dist)
            
        return ans