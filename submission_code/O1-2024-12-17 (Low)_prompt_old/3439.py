class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        from collections import deque
        
        # Helper to build adjacency list
        def build_graph(n, edges):
            graph = [[] for _ in range(n)]
            for a, b in edges:
                graph[a].append(b)
                graph[b].append(a)
            return graph
        
        # Helper to find the diameter of a tree given adjacency
        # We'll do a BFS from any node to find the farthest node X,
        # then BFS from X to find the farthest node Y and the distance is the diameter.
        def find_diameter(adj):
            n = len(adj)
            if n == 0:
                return 0
            
            def bfs(start):
                visited = [-1]*n
                visited[start] = 0
                queue = deque([start])
                farthest_node = start
                while queue:
                    node = queue.popleft()
                    for nxt in adj[node]:
                        if visited[nxt] == -1:
                            visited[nxt] = visited[node] + 1
                            farthest_node = nxt
                            queue.append(nxt)
                return farthest_node, visited[farthest_node]
            
            # First BFS from node 0 (or any node) to find one far end (X)
            x, _ = bfs(0)
            # Second BFS from X to find the actual diameter
            y, diameter = bfs(x)
            return diameter
        
        # Get the number of nodes for the two trees
        # They can be inferred from the edges arrays
        # The problem states they are "valid trees"
        # n = max node index in edges1 + 1
        # m = max node index in edges2 + 1
        # but we only need adjacency to find diameter, so let's do that:
        if not edges1:
            # If edges1 is empty, it means there's just 1 node in the first tree
            n = 1
        else:
            n = max(max(a, b) for a,b in edges1) + 1
        
        if not edges2:
            m = 1
        else:
            m = max(max(a, b) for a,b in edges2) + 1
        
        # Build adjacency lists
        adj1 = build_graph(n, edges1)
        adj2 = build_graph(m, edges2)
        
        # Find diameters
        d1 = find_diameter(adj1)
        d2 = find_diameter(adj2)
        
        # Compute radii
        r1 = (d1 + 1) // 2
        r2 = (d2 + 1) // 2
        
        # The new diameter after connecting one node from the first tree
        # to one node from the second tree by one edge
        # is given by: max(d1, d2, r1 + r2 + 1)
        return max(d1, d2, r1 + r2 + 1)