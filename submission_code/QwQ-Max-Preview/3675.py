from typing import List
from collections import deque

class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        n = len(edges) + 1
        adj = [[] for _ in range(n)]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        
        # Build children list using BFS
        children = [[] for _ in range(n)]
        parent = [-1] * n
        q = deque([0])
        parent[0] = -1
        while q:
            u = q.popleft()
            for v, w in adj[u]:
                if v != parent[u]:
                    if parent[v] == -1:
                        parent[v] = u
                        children[u].append((v, w))
                        q.append(v)
        
        take = [0] * n
        not_take = [0] * n
        
        stack = [(0, False)]
        while stack:
            u, visited = stack.pop()
            if not visited:
                stack.append((u, True))
                # Push children in reverse order to process them in order
                for v, w in reversed(children[u]):
                    stack.append((v, False))
            else:
                sum_not_take_children = 0
                deltas = []
                for v, w_uv in children[u]:
                    sum_not_take_children += not_take[v]
                    delta = take[v] + w_uv - not_take[v]
                    deltas.append(delta)
                # Sort deltas in descending order
                deltas.sort(reverse=True)
                # Compute prefix sums of positive deltas
                prefix = []
                current_sum = 0
                for delta in deltas:
                    if delta <= 0:
                        break
                    current_sum += delta
                    prefix.append(current_sum)
                # Compute not_take_u
                t_not_take = min(k, len(prefix))
                sum_deltas_not = prefix[t_not_take - 1] if t_not_take > 0 else 0
                not_take[u] = sum_not_take_children + sum_deltas_not
                # Compute take_u
                t_take = min(k - 1, len(prefix))
                sum_deltas_take = prefix[t_take - 1] if t_take > 0 else 0
                take[u] = sum_not_take_children + sum_deltas_take
        
        return not_take[0]