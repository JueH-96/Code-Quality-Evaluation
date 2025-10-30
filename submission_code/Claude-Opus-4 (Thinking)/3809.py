class Solution:
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        n = len(properties)
        
        # Function to find intersection count
        def intersect(a, b):
            return len(set(a) & set(b))
        
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if intersect(properties[i], properties[j]) >= k:
                    adj[i].append(j)
                    adj[j].append(i)
        
        # Count connected components using DFS
        visited = [False] * n
        components = 0
        
        def dfs(node):
            visited[node] = True
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    dfs(neighbor)
        
        for i in range(n):
            if not visited[i]:
                dfs(i)
                components += 1
        
        return components