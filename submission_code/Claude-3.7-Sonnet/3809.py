class Solution:
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        n = len(properties)
        
        # Convert properties to sets for efficient intersection
        property_sets = [set(prop) for prop in properties]
        
        # Build the graph based on intersection size
        graph = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if len(property_sets[i].intersection(property_sets[j])) >= k:
                    graph[i].append(j)
                    graph[j].append(i)
        
        # Count connected components using DFS
        visited = [False] * n
        
        def dfs(node):
            visited[node] = True
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    dfs(neighbor)
        
        connected_components = 0
        for i in range(n):
            if not visited[i]:
                dfs(i)
                connected_components += 1
        
        return connected_components