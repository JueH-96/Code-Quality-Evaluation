from typing import List
from collections import defaultdict

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        n = len(edges1) + 1
        m = len(edges2) + 1
        graph1 = defaultdict(list)
        graph2 = defaultdict(list)
        for u, v in edges1:
            graph1[u].append(v)
            graph1[v].append(u)
        for u, v in edges2:
            graph2[u].append(v)
            graph2[v].append(u)
        max_depth1 = [0] * n
        max_depth2 = [0] * m
        def dfs(node, parent, depth, graph, max_depth):
            max_depth[node] = depth
            for nei in graph[node]:
                if nei != parent:
                    dfs(nei, node, depth + 1, graph, max_depth)
        dfs(0, -1, 0, graph1, max_depth1)
        dfs(0, -1, 0, graph2, max_depth2)
        max_depth1_node = max_depth1.index(max(max_depth1))
        max_depth2_node = max_depth2.index(max(max_depth2))
        max_depth1[max_depth1_node] = 0
        max_depth2[max_depth2_node] = 0
        dfs(max_depth1_node, -1, 0, graph1, max_depth1)
        dfs(max_depth2_node, -1, 0, graph2, max_depth2)
        return max(max(max_depth1), max(max_depth2))