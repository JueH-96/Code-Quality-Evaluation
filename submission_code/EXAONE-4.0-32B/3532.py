import sys
sys.setrecursionlimit(300000)

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        B = [1 if i % 2 == 1 else 2 for i in range(n)]
        
        graph = [[] for _ in range(n)]
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
            
        parent = [-1] * n
        children = [[] for _ in range(n)]
        stack = [0]
        parent[0] = -1
        while stack:
            u = stack.pop()
            for v in graph[u]:
                if v == parent[u]:
                    continue
                parent[v] = u
                children[u].append(v)
                stack.append(v)
                
        order = []
        stack = [0]
        while stack:
            u = stack.pop()
            order.append(u)
            for v in children[u]:
                stack.append(v)
                
        dp1 = [0] * n
        for i in range(len(order)-1, -1, -1):
            u = order[i]
            candidates = []
            for v in children[u]:
                candidates.append(B[v] + dp1[v])
            if candidates:
                dp1[u] = max(candidates)
                
        best1 = [-10**18] * n
        best2 = [-10**18] * n
        for u in range(n):
            values = []
            for v in children[u]:
                values.append(B[v] + dp1[v])
            if not values:
                continue
            if len(values) == 1:
                best1[u] = values[0]
            else:
                values.sort(reverse=True)
                best1[u] = values[0]
                best2[u] = values[1]
                
        dp2 = [0] * n
        stack = [0]
        while stack:
            u = stack.pop()
            for v in children[u]:
                if B[v] + dp1[v] == best1[u]:
                    candidate = best2[u]
                else:
                    candidate = best1[u]
                candidate_value = max(dp2[u], candidate, 0)
                dp2[v] = B[u] + candidate_value
                stack.append(v)
                
        res = [0] * n
        for u in range(n):
            res[u] = max(dp1[u], dp2[u])
            
        return res