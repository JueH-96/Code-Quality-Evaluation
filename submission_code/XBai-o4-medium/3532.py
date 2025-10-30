from typing import List

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        up = [0] * n
        first_max_val = [0] * n
        second_max_val = [0] * n
        
        # Post-order traversal to compute up, first_max_val, second_max_val
        stack = [(0, -1, False)]
        while stack:
            u, parent_u, visited = stack.pop()
            if not visited:
                stack.append((u, parent_u, True))
                # Push children to process
                for v in adj[u]:
                    if v != parent_u:
                        stack.append((v, u, False))
            else:
                first = -float('inf')
                second = -float('inf')
                for v in adj[u]:
                    if v != parent_u:
                        if v % 2 == 1:
                            contrib = 1 + up[v]
                        else:
                            contrib = 2 + up[v]
                        if contrib > first:
                            second = first
                            first = contrib
                        elif contrib > second:
                            second = contrib
                # Compute first_max_val and second_max_val
                if first == -float('inf'):
                    first_max_val[u] = 0
                    second_max_val[u] = 0
                else:
                    first_max_val[u] = first
                    if second == -float('inf'):
                        second_max_val[u] = 0
                    else:
                        second_max_val[u] = second
                up[u] = first_max_val[u]
        
        ans = [0] * n
        ans[0] = up[0]
        # Pre-order traversal to compute down values and ans
        stack = [(0, -1, 0)]
        while stack:
            u, parent_u, down_u = stack.pop()
            for v in adj[u]:
                if v != parent_u:
                    # Calculate contribution_v
                    if v % 2 == 1:
                        contrib_v = 1 + up[v]
                    else:
                        contrib_v = 2 + up[v]
                    # Check if contrib_v equals first_max_val[u]
                    if contrib_v == first_max_val[u]:
                        best_from_children = second_max_val[u]
                    else:
                        best_from_children = first_max_val[u]
                    best_from_u = max(best_from_children, down_u)
                    # Compute weight from v to u based on u's parity
                    if u % 2 == 1:
                        weight_v_to_u = 1
                    else:
                        weight_v_to_u = 2
                    new_down_v = weight_v_to_u + best_from_u
                    # Update ans[v]
                    ans[v] = max(up[v], new_down_v)
                    # Push to stack
                    stack.append((v, u, new_down_v))
        
        return ans