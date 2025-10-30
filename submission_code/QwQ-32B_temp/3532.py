class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        import sys
        n = len(edges) + 1
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        INF = float('inf')
        
        # Initialize arrays for f, max1, max2
        f = [0] * n
        max1 = [(-INF, -1) for _ in range(n)]
        max2 = [(-INF, -1) for _ in range(n)]
        
        # Post-order traversal to compute f, max1, max2
        stack = []
        processed = [False] * n
        stack.append((0, -1, False))
        
        while stack:
            node, parent_node, visited = stack.pop()
            if not visited:
                stack.append((node, parent_node, True))
                # Push children (excluding parent_node) in reverse order
                for neighbor in adj[node]:
                    if neighbor != parent_node:
                        stack.append((neighbor, node, False))
            else:
                children = []
                for neighbor in adj[node]:
                    if neighbor != parent_node:
                        children.append(neighbor)
                
                current_max1 = -INF
                current_max2 = -INF
                best_child1 = -1
                best_child2 = -1
                
                for v in children:
                    contrib = 1 if (v % 2) else 2
                    val = contrib + f[v]
                    if val > current_max1:
                        current_max2 = current_max1
                        current_max1 = val
                        best_child2 = best_child1
                        best_child1 = v
                    elif val > current_max2:
                        current_max2 = val
                        best_child2 = v
                
                max1[node] = (current_max1, best_child1)
                max2[node] = (current_max2, best_child2)
                
                if children:
                    f[node] = current_max1
                else:
                    f[node] = 0
        
        ans = f.copy()
        
        from collections import deque
        q = deque()
        q.append((0, -1))
        
        while q:
            u, p = q.popleft()
            for v in adj[u]:
                if v != p:
                    # Compute candidate for v
                    contrib_vu = 1 if (u % 2) else 2
                    m1, c1 = max1[u]
                    m2, c2 = max2[u]
                    
                    if v == c1:
                        candidate_max = m2
                    elif v == c2:
                        candidate_max = m1
                    else:
                        candidate_max = m1
                    
                    max_without_v = max(candidate_max, 0)
                    candidate = contrib_vu + max_without_v
                    
                    if candidate > ans[v]:
                        ans[v] = candidate
                    
                    q.append((v, u))
        
        return ans