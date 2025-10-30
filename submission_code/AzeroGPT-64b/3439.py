class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def dfs(node, parent, height):
            h = 0
            for neib in graph[node]:
                if neib != parent:
                    h = max(h, dfs(neib, node, height + 1))
            return max(height, h)
        
        n = len(edges1) + 1
        m = len(edges2) + 1

        graph1 = [[] for i in range(n)]
        graph2 = [[] for i in range(m)]

        for u, v in edges1:
            graph1[u].append(v)
            graph1[v].append(u)
        for u, v in edges2:
            graph2[u].append(v)
            graph2[v].append(u)
        
        max_d1, max_d2 = 0, 0
        for node in range(n):
            max_d1 = max(max_d1, dfs(node, -1, 0))
        for node in range(m):
            max_d2 = max(max_d2, dfs(node, -1, 0))

        return min(max_d1 + 1, max_d2 + 1)