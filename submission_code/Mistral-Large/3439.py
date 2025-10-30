from collections import deque, defaultdict
from typing import List

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def bfs(start, graph):
            visited = set()
            queue = deque([(start, 0)])
            max_dist = 0
            farthest_node = start

            while queue:
                node, dist = queue.popleft()
                if dist > max_dist:
                    max_dist = dist
                    farthest_node = node
                visited.add(node)
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        queue.append((neighbor, dist + 1))

            return farthest_node, max_dist

        def tree_diameter(edges):
            graph = defaultdict(list)
            for u, v in edges:
                graph[u].append(v)
                graph[v].append(u)

            # First BFS to find the farthest node from an arbitrary starting node
            start_node = 0
            farthest_node, _ = bfs(start_node, graph)

            # Second BFS to find the actual diameter
            _, diameter = bfs(farthest_node, graph)

            return diameter

        diameter1 = tree_diameter(edges1)
        diameter2 = tree_diameter(edges2)

        return max(diameter1, diameter2) + 1

# Example usage:
# edges1 = [[0,1],[0,2],[0,3]]
# edges2 = [[0,1]]
# solution = Solution()
# print(solution.minimumDiameterAfterMerge(edges1, edges2))  # Output: 3