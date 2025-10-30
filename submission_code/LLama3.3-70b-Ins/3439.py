from typing import List

class Solution:
    def treeDiameter(self, edges: List[List[int]], n: int) -> int:
        graph = [[] for _ in range(n)]
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        def bfs(start: int) -> int:
            queue = [(start, 0)]
            visited = {start}
            max_distance = 0
            max_node = start
            while queue:
                node, distance = queue.pop(0)
                if distance > max_distance:
                    max_distance = distance
                    max_node = node
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))
            return max_node, max_distance

        node1, _ = bfs(0)
        _, diameter = bfs(node1)
        return diameter

    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        n, m = len(edges1) + 1, len(edges2) + 1
        diameter1 = self.treeDiameter(edges1, n)
        diameter2 = self.treeDiameter(edges2, m)
        
        # Connect the two trees at their centers
        return max(diameter1, diameter2)