class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def get_diameter(edges):
            n = len(edges) + 1
            graph = [[] for _ in range(n)]
            for u, v in edges:
                graph[u].append(v)
                graph[v].append(u)

            def bfs(start_node):
                queue = [(start_node, 0)]
                visited = {start_node}
                farthest_node = start_node
                max_dist = 0
                while queue:
                    curr_node, curr_dist = queue.pop(0)
                    if curr_dist > max_dist:
                        max_dist = curr_dist
                        farthest_node = curr_node
                    for neighbor in graph[curr_node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, curr_dist + 1))
                return farthest_node, max_dist

            farthest_node1, _ = bfs(0)
            farthest_node2, diameter = bfs(farthest_node1)
            return diameter

        diameter1 = get_diameter(edges1)
        diameter2 = get_diameter(edges2)

        min_diameter = float('inf')
        for i in range(len(edges1) + 1):
            for j in range(len(edges2) + 1):
                new_edges = edges1 + edges2 + [[i, len(edges1) + j]]
                min_diameter = min(min_diameter, get_diameter(new_edges))

        return min_diameter