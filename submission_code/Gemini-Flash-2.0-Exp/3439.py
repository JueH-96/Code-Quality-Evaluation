from collections import defaultdict

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def get_diameter(edges, n):
            adj = defaultdict(list)
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)

            def bfs(start_node):
                q = [(start_node, 0)]
                visited = {start_node}
                max_dist = 0
                farthest_node = start_node

                while q:
                    node, dist = q.pop(0)
                    if dist > max_dist:
                        max_dist = dist
                        farthest_node = node

                    for neighbor in adj[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append((neighbor, dist + 1))

                return farthest_node, max_dist

            node1, _ = bfs(0)
            node2, diameter = bfs(node1)
            return diameter

        diameter1 = get_diameter(edges1, len(edges1) + 1)
        diameter2 = get_diameter(edges2, len(edges2) + 1)

        return max((diameter1 + 1) // 2 + (diameter2 + 1) // 2 + 1, diameter1, diameter2)