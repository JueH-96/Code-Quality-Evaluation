class Solution:
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        from collections import defaultdict, deque
        
        def intersect(a, b):
            return len(set(a) & set(b))
        
        n = len(properties)
        graph = defaultdict(list)
        
        # Build the graph
        for i in range(n):
            for j in range(i + 1, n):
                if intersect(properties[i], properties[j]) >= k:
                    graph[i].append(j)
                    graph[j].append(i)
        
        visited = [False] * n
        component_count = 0
        
        # Helper function to perform DFS
        def dfs(node):
            stack = [node]
            while stack:
                current = stack.pop()
                for neighbor in graph[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
        
        # Count connected components using DFS
        for i in range(n):
            if not visited[i]:
                visited[i] = True
                dfs(i)
                component_count += 1
        
        return component_count