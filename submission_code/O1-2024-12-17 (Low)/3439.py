class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        from collections import deque
        
        # Build adjacency from edge list
        def build_adj(n, edges):
            adj = [[] for _ in range(n)]
            for a,b in edges:
                adj[a].append(b)
                adj[b].append(a)
            return adj
        
        # Given an adjacency list, run BFS from a start node to find the farthest node and distance
        def bfs_farthest(adj, start):
            n = len(adj)
            visited = [False] * n
            queue = deque([(start, 0)])
            visited[start] = True
            farthest_node = start
            max_dist = 0
            
            while queue:
                node, dist = queue.popleft()
                if dist > max_dist:
                    max_dist = dist
                    farthest_node = node
                for nxt in adj[node]:
                    if not visited[nxt]:
                        visited[nxt] = True
                        queue.append((nxt, dist + 1))
            return farthest_node, max_dist
        
        # Compute diameter of a tree using 2 BFS
        def get_diameter(edges):
            n = len(edges) + 1
            if n == 1:
                return 0  # single node => diameter 0
            adj = build_adj(n, edges)
            # First BFS from node 0 (or any node, tree is connected)
            farthest, _ = bfs_farthest(adj, 0)
            # Second BFS from 'farthest' found
            _, diameter = bfs_farthest(adj, farthest)
            return diameter
        
        diam1 = get_diameter(edges1)
        diam2 = get_diameter(edges2)
        
        # Radius of a tree = (diameter + 1) // 2
        r1 = (diam1 + 1) // 2
        r2 = (diam2 + 1) // 2
        
        # Min possible diameter = max(diam1, diam2, r1 + r2 + 1)
        return max(diam1, diam2, r1 + r2 + 1)