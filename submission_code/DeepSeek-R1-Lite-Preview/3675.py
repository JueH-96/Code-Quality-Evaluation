class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        from collections import defaultdict, deque

        n = len(edges) + 1
        adj = defaultdict(list)
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))

        dp0 = [0] * n  # dp[u][0]: u not connected to parent
        dp1 = [0] * n  # dp[u][1]: u connected to parent

        # Iterative post-order DFS using stack
        stack = []
        visited = [False] * n
        stack.append((0, -1))  # (node, parent)
        
        # To temporarily store child contributions
        child_contrib = [ [] for _ in range(n) ]

        # First pass to collect children
        while stack:
            u, parent = stack.pop()
            if not visited[u]:
                visited[u] = True
                stack.append((u, parent, True))  # marker for processing after children
                for v, w in adj[u]:
                    if v != parent:
                        stack.append((v, u))
            else:
                children = [v for v, w in adj[u] if v != parent]
                child_contrib[u] = []
                for v, w in adj[u]:
                    if v != parent and visited[v]:
                        child_contrib[u].append((w, v))
                # Now sort the child edges by weight descending
                child_contrib[u].sort(reverse=True, key=lambda x: x[0])
                
        # Now compute dp0 and dp1
        stack = []
        visited = [False] * n
        stack.append((0, -1))  # (node, parent)
        while stack:
            u, parent = stack.pop()
            if not visited[u]:
                visited[u] = True
                has_children = False
                for v, w in adj[u]:
                    if v != parent:
                        has_children = True
                        stack.append((u, parent, True))  # marker for processing after children
                        stack.append((v, u))
                        break
                # Push remaining children
                for v, w in adj[u]:
                    if v != parent and v != stack[-1][0] if stack else False:
                        stack.append((v, u))
            else:
                # Processing node u after its children
                children = [v for v, w in adj[u] if v != parent]
                m0 = min(k, len(children))
                m1 = min(k-1, len(children)) if k >= 1 else 0

                # For dp0[u]
                sum_dp0 = 0
                top_m0_sum = 0
                top_m0_selected = []
                for i in range(len(child_contrib[u])):
                    w, v = child_contrib[u][i]
                    if i < m0:
                        top_m0_sum += w + dp1[v]
                        top_m0_selected.append(v)
                    else:
                        sum_dp0 += dp0[v]
                dp0[u] = top_m0_sum + sum_dp0

                # For dp1[u]
                if k >= 1:
                    sum_dp1 = 0
                    top_m1_sum = 0
                    top_m1_selected = []
                    for i in range(len(child_contrib[u])):
                        w, v = child_contrib[u][i]
                        if i < m1:
                            top_m1_sum += w + dp1[v]
                            top_m1_selected.append(v)
                        else:
                            sum_dp1 += dp0[v]
                    dp1[u] = top_m1_sum + sum_dp1
                else:
                    dp1[u] = 0  # Cannot connect to any children if k=0

        return dp0[0]