from typing import List
from collections import deque

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def bfs(tree, start):
            queue = deque([(start, 0)])
            visited = set()
            max_distance = 0
            farthest_node = start

            while queue:
                node, distance = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                if distance > max_distance:
                    max_distance = distance
                    farthest_node = node
                for neighbor in tree[node]:
                    if neighbor not in visited:
                        queue.append((neighbor, distance + 1))
            return farthest_node, max_distance

        def build_tree(edges, n):
            tree = {i: [] for i in range(n)}
            for a, b in edges:
                tree[a].append(b)
                tree[b].append(a)
            return tree

        n = len(edges1) + 1
        m = len(edges2) + 1

        tree1 = build_tree(edges1, n)
        tree2 = build_tree(edges2, m)

        # Find the diameter of tree1
        farthest_node1, _ = bfs(tree1, 0)
        _, diameter1 = bfs(tree1, farthest_node1)

        # Find the diameter of tree2
        farthest_node2, _ = bfs(tree2, 0)
        _, diameter2 = bfs(tree2, farthest_node2)

        # The minimum possible diameter after merging the two trees
        return max(diameter1, diameter2)