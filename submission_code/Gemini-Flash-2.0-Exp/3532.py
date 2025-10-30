class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        def bfs(start_node):
            times = [-1] * n
            times[start_node] = 0
            q = [(start_node, 0)]
            
            while q:
                node, time = q.pop(0)
                
                for neighbor in adj[node]:
                    if times[neighbor] == -1:
                        if neighbor % 2 == 1:
                            times[neighbor] = time + 1
                            q.append((neighbor, time + 1))
                        else:
                            times[neighbor] = time + 2
                            q.append((neighbor, time + 2))
            
            return max(times)
        
        result = []
        for i in range(n):
            result.append(bfs(i))
        
        return result