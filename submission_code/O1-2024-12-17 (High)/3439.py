class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        from collections import deque

        # Helper function to run BFS and return the distance array from a given start node
        def bfs(adj, start):
            dist = [-1] * len(adj)
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            return dist
        
        # Computes the diameter of the tree represented by adjacency list 'adj'
        def tree_diameter(adj):
            # 1) BFS from an arbitrary node (0) to find the farthest node p1
            dist0 = bfs(adj, 0)
            p1 = max(range(len(adj)), key=lambda x: dist0[x])
            
            # 2) BFS from p1 to find the farthest node p2 and the distance to it (the diameter)
            dist1 = bfs(adj, p1)
            p2 = max(range(len(adj)), key=lambda x: dist1[x])
            
            return dist1[p2]  # distance between p1 and p2 is the diameter

        n = len(edges1) + 1  # Number of nodes in the first tree
        m = len(edges2) + 1  # Number of nodes in the second tree

        # Build adjacency lists for both trees
        adj1 = [[] for _ in range(n)]
        for a, b in edges1:
            adj1[a].append(b)
            adj1[b].append(a)

        adj2 = [[] for _ in range(m)]
        for u, v in edges2:
            adj2[u].append(v)
            adj2[v].append(u)

        # Compute diameters of both trees
        dia1 = tree_diameter(adj1)
        dia2 = tree_diameter(adj2)

        # The radius of a tree of diameter d is (d + 1) // 2
        r1 = (dia1 + 1) // 2
        r2 = (dia2 + 1) // 2

        # Once we connect a node from the first tree and a node from the second tree,
        # the diameter of the merged tree is the maximum of:
        #   1) diameter of T1
        #   2) diameter of T2
        #   3) (r1 + r2 + 1)
        # The minimal possible diameter after an optimal merge:
        return max(dia1, dia2, r1 + r2 + 1)