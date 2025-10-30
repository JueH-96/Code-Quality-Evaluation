from typing import List

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        n = len(edges1)
        m = len(edges2)
        graph1 = [[] for _ in range(n)]
        graph2 = [[] for _ in range(m)]
        for u, v in edges1:
            graph1[u].append(v)
            graph1[v].append(u)
        for u, v in edges2:
            graph2[u].append(v)
            graph2[v].append(u)
        diameter = [0]
        def dfs(node, parent):
            d1 = d2 = 0
            for child in graph1[node]:
                if child == parent:
                    continue
                d = dfs(child, node)
                if d > d1:
                    d1, d2 = d, d1
                elif d > d2:
                    d2 = d
            diameter[0] = max(diameter[0], d1 + d2)
            return d1 + 1
        dfs(0, -1)
        return diameter[0]