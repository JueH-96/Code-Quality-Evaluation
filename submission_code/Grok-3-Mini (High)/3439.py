import collections
from typing import List

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def bfs(adj, start):
            n = len(adj)
            dist = [-1] * n
            dist[start] = 0
            queue = collections.deque([start])
            while queue:
                curr = queue.popleft()
                for nei in adj[curr]:
                    if dist[nei] == -1:
                        dist[nei] = dist[curr] + 1
                        queue.append(nei)
            # Find max distance and the node achieving it
            max_dist = 0
            max_node = start
            for i in range(n):
                if dist[i] > max_dist:
                    max_dist = dist[i]
                    max_node = i
            return max_dist, max_node
        
        def get_diameter(edges):
            num_nodes = len(edges) + 1
            adj = [[] for _ in range(num_nodes)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            # BFS from node 0 to find a farthest node
            _, node_u = bfs(adj, 0)
            # BFS from node_u to find the diameter
            diam, _ = bfs(adj, node_u)
            return diam
        
        # Compute diameters
        D_A = get_diameter(edges1)
        D_B = get_diameter(edges2)
        
        # Compute radii using ceiling of diameter / 2
        R_A = (D_A + 1) // 2
        R_B = (D_B + 1) // 2
        
        # M is the maximum of the two diameters
        M = max(D_A, D_B)
        
        # Check the condition to minimize the new diameter
        if R_A + R_B <= M - 1:
            return M
        else:
            return R_A + R_B + 1