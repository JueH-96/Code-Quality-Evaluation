from typing import List

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        max_gain = [0] * n
        max1 = [0] * n
        max2 = [0] * n
        parent = [-1] * n
        
        # Post-order traversal to compute max1 and max2
        stack = [(0, -1, False)]
        post_stack = []
        while stack:
            u, p, visited = stack.pop()
            if not visited:
                parent[u] = p
                post_stack.append((u, p))
                stack.append((u, p, True))
                for v in adj[u]:
                    if v != p:
                        stack.append((v, u, False))
            else:
                gains = []
                for v in adj[u]:
                    if v != parent[u]:
                        weight = 1 if (v % 2 == 1) else 2
                        gains.append(weight + max_gain[v])
                max1_u, max2_u = 0, 0
                for g in gains:
                    if g > max1_u:
                        max2_u = max1_u
                        max1_u = g
                    elif g > max2_u:
                        max2_u = g
                max1[u] = max1_u
                max2[u] = max2_u
                max_gain[u] = max1_u
        
        # Pre-order traversal to update max_gain for all nodes
        stack = [(0, -1)]
        while stack:
            u, p = stack.pop()
            for v in adj[u]:
                if v == parent[u]:
                    continue
                weight_u_v = 1 if (v % 2 == 1) else 2
                gain_from_v = weight_u_v + max_gain[v]
                if gain_from_v == max1[u]:
                    current_max = max2[u]
                else:
                    current_max = max1[u]
                weight_v_u = 1 if (u % 2 == 1) else 2
                new_gain = weight_v_u + current_max
                if new_gain > max_gain[v]:
                    max_gain[v] = new_gain
                stack.append((v, u))
        
        return max_gain