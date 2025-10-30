from typing import List
from collections import deque

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initialize down, max1_down, max2_down arrays
        down = [0] * n
        max1_down = [0] * n
        max2_down = [0] * n
        
        # Post-order traversal to compute down values and max1/max2 for children
        stack = [(0, -1, False)]
        while stack:
            u, parent, visited = stack.pop()
            if not visited:
                stack.append((u, parent, True))
                for v in adj[u]:
                    if v != parent:
                        stack.append((v, u, False))
            else:
                max1 = 0
                max2 = 0
                for v in adj[u]:
                    if v != parent:
                        contribution = 1 if v % 2 == 1 else 2
                        current_sum = contribution + down[v]
                        if current_sum > max1:
                            max2 = max1
                            max1 = current_sum
                        elif current_sum > max2:
                            max2 = current_sum
                down[u] = max1
                max1_down[u] = max1
                max2_down[u] = max2
        
        # Pre-order traversal to compute up values
        up = [0] * n
        q = deque()
        q.append((0, -1))
        while q:
            u, parent = q.popleft()
            for v in adj[u]:
                if v == parent:
                    continue
                # Calculate contribution from u to v (v's contribution is based on u's parity)
                contribution = 1 if u % 2 == 1 else 2
                current_sum_v = (1 if v % 2 == 1 else 2) + down[v]
                if current_sum_v == max1_down[u]:
                    other_max_down = max2_down[u]
                else:
                    other_max_down = max1_down[u]
                other_max = max(other_max_down, up[u])
                up[v] = contribution + other_max
                q.append((v, u))
        
        # Calculate the answer for each node
        answer = [0] * n
        for i in range(n):
            answer[i] = max(down[i], up[i])
        
        return answer