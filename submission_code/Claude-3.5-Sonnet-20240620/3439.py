class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def build_graph(edges):
            graph = {}
            for u, v in edges:
                graph.setdefault(u, []).append(v)
                graph.setdefault(v, []).append(u)
            return graph

        def dfs(node, parent, graph):
            max_depth = 0
            second_max_depth = 0
            diameter = 0
            
            for child in graph[node]:
                if child != parent:
                    child_depth, child_diameter = dfs(child, node, graph)
                    diameter = max(diameter, child_diameter)
                    if child_depth > max_depth:
                        second_max_depth = max_depth
                        max_depth = child_depth
                    elif child_depth > second_max_depth:
                        second_max_depth = child_depth
            
            diameter = max(diameter, max_depth + second_max_depth)
            return max_depth + 1, diameter

        graph1 = build_graph(edges1)
        graph2 = build_graph(edges2)

        _, diameter1 = dfs(0, -1, graph1)
        _, diameter2 = dfs(0, -1, graph2)

        max_depth1 = max(dfs(node, -1, graph1)[0] for node in graph1)
        max_depth2 = max(dfs(node, -1, graph2)[0] for node in graph2)

        return max(diameter1, diameter2, max_depth1 + max_depth2 - 1)