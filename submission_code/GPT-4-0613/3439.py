from typing import List
from collections import defaultdict

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def get_diameter(edges):
            graph = defaultdict(list)
            for u, v in edges:
                graph[u].append(v)
                graph[v].append(u)
            def dfs(node, parent):
                depths = [dfs(child, node) for child in graph[node] if child != parent]
                depths.sort()
                if len(depths) >= 2:
                    self.longest_path = max(self.longest_path, depths[-1] + depths[-2] + 2)
                return (depths[-1] + 1) if depths else 0
            self.longest_path = 0
            dfs(0, None)
            return self.longest_path
        d1 = get_diameter(edges1)
        d2 = get_diameter(edges2)
        return max(d1, d2, (d1 + 1) // 2 + (d2 + 1) // 2 + 1)