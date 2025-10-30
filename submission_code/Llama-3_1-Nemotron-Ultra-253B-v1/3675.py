from typing import List
from collections import deque

class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        n = len(edges) + 1  # Number of nodes
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        
        # Build the tree structure with parent pointers and children list
        parent = [-1] * n
        children = [[] for _ in range(n)]  # Each entry is a list of (child, weight)
        q = deque([0])
        parent[0] = -1
        while q:
            u = q.popleft()
            for v, w in adj[u]:
                if parent[v] == -1 and v != parent[u]:
                    parent[v] = u
                    children[u].append((v, w))
                    q.append(v)
        
        # Post-order traversal using stack
        post_order = []
        stack = [(0, False)]
        while stack:
            u, visited = stack.pop()
            if visited:
                post_order.append(u)
                continue
            stack.append((u, True))
            # Push children in reverse order to process them left to right
            for v, w in reversed(children[u]):
                stack.append((v, False))
        
        # Initialize DP arrays
        dp_keep = [0] * n
        dp_not_keep = [0] * n
        
        for u in post_order:
            a_list = []
            b_list = []
            for v, w in children[u]:
                a = w + dp_keep[v]
                b = dp_not_keep[v]
                a_list.append(a)
                b_list.append(b)
            differences = [a - b for a, b in zip(a_list, b_list)]
            differences.sort(reverse=True)
            sum_b = sum(b_list)
            
            # Compute dp_keep[u] (allowed to take up to k-1 edges)
            m = k - 1
            sum_keep = sum_b
            for diff in differences:
                if diff > 0 and m > 0:
                    sum_keep += diff
                    m -= 1
                else:
                    break
            dp_keep[u] = sum_keep
            
            # Compute dp_not_keep[u] (allowed to take up to k edges)
            m = k
            sum_not_keep = sum_b
            for diff in differences:
                if diff > 0 and m > 0:
                    sum_not_keep += diff
                    m -= 1
                else:
                    break
            dp_not_keep[u] = sum_not_keep
        
        return dp_not_keep[0]