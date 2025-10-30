class Solution:
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        n = len(properties)
        
        # Helper function to find intersection count
        def intersect(a, b):
            # Convert to sets and find common elements
            set_a = set(a)
            set_b = set(b)
            return len(set_a & set_b)
        
        # Build adjacency list for the graph
        adj = [[] for _ in range(n)]
        
        # Check all pairs of nodes
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
        
        # Find all connected components
        for i in range(n):
            if not visited[i]:
                dfs(i)
                components += 1
        
        return components