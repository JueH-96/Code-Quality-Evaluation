from collections import deque, defaultdict
from typing import List

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        n = len(edges1) + 1
        m = len(edges2) + 1

        def get_diameter(num_nodes: int, edges: List[List[int]]) -> int:
            """Calculates the diameter of a tree."""
            if num_nodes <= 1:
                return 0

            # Build adjacency list
            adj = defaultdict(list)
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)

            def bfs(start_node: int):
                """Performs BFS to find the farthest node and its distance from start_node."""
                distances = [-1] * num_nodes
                q = deque([(start_node, 0)])
                distances[start_node] = 0
                
                farthest_node = start_node
                max_dist = 0

                while q:
                    curr_node, dist = q.popleft()

                    if dist > max_dist:
                        max_dist = dist
                        farthest_node = curr_node

                    # Iterate over neighbors. Node indices are 0 to num_nodes-1.
                    # Since it's a tree with num_nodes > 1, all nodes are connected and
                    # will eventually be visited starting from 0 (or any node).
                    for neighbor in adj[curr_node]:
                        if distances[neighbor] == -1:
                            distances[neighbor] = dist + 1
                            q.append((neighbor, dist + 1))
                
                return farthest_node, max_dist

            # The diameter of a tree can be found by:
            # 1. Starting a BFS from an arbitrary node (e.g., node 0) to find the farthest node (endpoint1).
            # 2. Starting a second BFS from endpoint1 to find the farthest node from it.
            #    The distance found in the second BFS is the diameter.

            # Find one endpoint of a diameter by BFS from node 0 (valid node index 0 to num_nodes-1)
            endpoint1, _ = bfs(0)

            # Find the other endpoint and the diameter by BFS from endpoint1
            _, diameter = bfs(endpoint1)

            return diameter

        # Calculate diameter of Tree 1
        D1 = get_diameter(n, edges1)

        # Calculate diameter of Tree 2
        D2 = get_diameter(m, edges2)

        # Calculate radius of Tree 1 and Tree 2
        # The radius R of a tree is ceil(Diameter / 2).
        # Using integer division: R = (D + 1) // 2
        R1 = (D1 + 1) // 2
        R2 = (D2 + 1) // 2

        # When we merge the two trees with an edge between node u (from T1) and node v (from T2),
        # the diameter of the new tree is the maximum of:
        # 1. The diameter of Tree 1 (paths entirely within T1).
        # 2. The diameter of Tree 2 (paths entirely within T2).
        # 3. The length of the longest path that crosses the new edge (u, v).
        #    A path crossing (u, v) goes from some node x in T1 to some node y in T2,
        #    passing through u and v: x -> ... -> u -> v -> ... -> y.
        #    The length is dist1(x, u) + 1 + dist2(v, y).
        #    For fixed u and v, the maximum such length is max_dist1(u) + 1 + max_dist2(v),
        #    where max_dist1(u) is the max distance from u to any node in T1, and
        #    max_dist2(v) is the max distance from v to any node in T2.
        #    To minimize the diameter of the merged tree, we need to minimize this
        #    maximum crossing path length by choosing the optimal u and v.
        #    The minimum value of max_dist1(u) over all u in T1 is the radius R1.
        #    The minimum value of max_dist2(v) over all v in T2 is the radius R2.
        #    Thus, the minimum possible length of a path crossing the new edge is R1 + 1 + R2,
        #    achieved by connecting a center of Tree 1 to a center of Tree 2.
        
        min_crossing_path_length = R1 + R2 + 1

        # The minimum diameter of the merged tree is the minimum possible value
        # of max(D1, D2, max_dist1(u) + 1 + max_dist2(v)) over all u, v.
        # This minimum is achieved when max_dist1(u) + max_dist2(v) is minimized (equal to R1 + R2).
        # So the minimum diameter is max(D1, D2, R1 + R2 + 1).
        
        return max(D1, D2, min_crossing_path_length)