from collections import defaultdict

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def dfs(node, parent, graph):
            max_depth = 0
            for neighbor in graph[node]:
                if neighbor != parent:
                    depth = 1 + dfs(neighbor, node, graph)
                    max_depth = max(max_depth, depth)
            return max_depth

        def diameter(graph):
            max_diameter = 0
            for node in range(len(graph)):
                depth1 = dfs(node, -1, graph)
                for neighbor in graph[node]:
                    depth2 = dfs(neighbor, node, graph)
                    max_diameter = max(max_diameter, depth1 + depth2)
            return max_diameter

        graph1 = defaultdict(list)
        for u, v in edges1:
            graph1[u].append(v)
            graph1[v].append(u)

        graph2 = defaultdict(list)
        for u, v in edges2:
            graph2[u].append(v)
            graph2[v].append(u)

        diameter1 = diameter(graph1)
        diameter2 = diameter(graph2)

        min_diameter = float('inf')
        for u in range(len(graph1)):
            for v in range(len(graph2)):
                new_diameter = max(diameter1, diameter2, dfs(u, -1, graph1) + dfs(v, -1, graph2) + 1)
                min_diameter = min(min_diameter, new_diameter)

        return min_diameter