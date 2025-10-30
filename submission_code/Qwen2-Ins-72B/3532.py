from collections import defaultdict

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def dfs(node, parent, depth):
            if depth % 2 == node % 2:
                return depth
            max_depth = depth
            for neighbor in graph[node]:
                if neighbor != parent:
                    max_depth = max(max_depth, dfs(neighbor, node, depth + 1))
            if max_depth % 2 == node % 2:
                return max_depth
            return max_depth + 1 if node % 2 == 0 else max_depth + 2
        
        return [dfs(i, -1, 0) for i in range(len(edges) + 1)]