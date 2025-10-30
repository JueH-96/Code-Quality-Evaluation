class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        # Build adjacency list
        graph = [[] for _ in range(n)]
        
        # Helper function to calculate LCM
        def lcm(a, b):
            from math import gcd
            return (a * b) // gcd(a, b)
        
        # Connect nodes if LCM <= threshold
        for i in range(n):
            for j in range(i+1, n):
                if lcm(nums[i], nums[j]) <= threshold:
                    graph[i].append(j)
                    graph[j].append(i)
        
        # Use DFS to find connected components
        visited = [False] * n
        component_count = 0
        
        def dfs(node):
            visited[node] = True
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    dfs(neighbor)
        
        # Count connected components
        for i in range(n):
            if not visited[i]:
                component_count += 1
                dfs(i)
        
        return component_count