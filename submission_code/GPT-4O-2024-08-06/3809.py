class Solution:
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        from collections import defaultdict
        
        def intersect(a, b):
            # Calculate the number of distinct integers common to both arrays a and b
            return len(set(a) & set(b))
        
        n = len(properties)
        
        # Create adjacency list for the graph
        graph = defaultdict(list)
        
        # Build the graph based on the intersection condition
        for i in range(n):
            for j in range(i + 1, n):
                if intersect(properties[i], properties[j]) >= k:
                    graph[i].append(j)
                    graph[j].append(i)
        
        # Function to perform DFS and mark all nodes in the same component
        def dfs(node, visited):
            stack = [node]
            while stack:
                current = stack.pop()
                for neighbor in graph[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
        
        visited = [False] * n
        components = 0
        
        # Count connected components using DFS
        for i in range(n):
            if not visited[i]:
                components += 1
                visited[i] = True
                dfs(i, visited)
        
        return components