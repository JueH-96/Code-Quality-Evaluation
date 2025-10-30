from collections import deque
from typing import List

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        # Helper function: perform BFS from a starting node.
        # It returns both the farthest node and the distance to that node.
        def bfs(start: int, graph: List[List[int]]) -> (int, int):
            n = len(graph)
            visited = [False] * n
            visited[start] = True
            q = deque([(start, 0)])
            farthest_node = start
            max_dist = 0
            while q:
                node, dist = q.popleft()
                if dist > max_dist:
                    max_dist = dist
                    farthest_node = node
                for neighbor in graph[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        q.append((neighbor, dist + 1))
            return farthest_node, max_dist

        # Helper function: compute the diameter of a tree.
        def get_diameter(edges: List[List[int]], num_nodes: int) -> int:
            if num_nodes <= 1:
                return 0
            # Build the adjacency list.
            graph = [[] for _ in range(num_nodes)]
            for u, v in edges:
                graph[u].append(v)
                graph[v].append(u)
            # Pick an arbitrary starting node (node 0) and find the farthest node.
            farthest_node, _ = bfs(0, graph)
            # Starting from that farthest node, find the maximum distance,
            # which is the diameter of the tree.
            _, diameter = bfs(farthest_node, graph)
            return diameter

        # The number of nodes is one more than the number of edges.
        n = len(edges1) + 1
        m = len(edges2) + 1
        d1 = get_diameter(edges1, n)
        d2 = get_diameter(edges2, m)
        
        # For a tree, the optimal connection minimizes the new diameter when
        # connecting the centers. The radius of a tree is defined as ceil(diameter/2),
        # which can be computed as (diameter + 1) // 2.
        r1 = (d1 + 1) // 2
        r2 = (d2 + 1) // 2
        
        # When we merge the two trees, the potential longest path could either be:
        # 1. Entirely in the first tree (diameter d1)
        # 2. Entirely in the second tree (diameter d2)
        # 3. A path that goes from one tree to the other, passing through both centers:
        #    This distance is r1 + r2 + 1 (1 for the new connecting edge).
        #
        # Thus, the minimum possible diameter is the maximum of these three values.
        return max(d1, d2, r1 + r2 + 1)