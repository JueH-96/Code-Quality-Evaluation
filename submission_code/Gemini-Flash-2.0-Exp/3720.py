class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
        def check(max_weight):
            adj = [[] for _ in range(n)]
            for u, v, w in edges:
                if w <= max_weight:
                    adj[u].append(v)

            for i in range(n):
                if len(adj[i]) > threshold:
                    return False

            q = [0]
            visited = [False] * n
            visited[0] = True
            
            reachable = [False] * n
            reachable[0] = True

            while q:
                u = q.pop(0)
                for v in range(n):
                    for a, b, w in edges:
                        if a == v and b == u and w <= max_weight:
                            reachable[v] = True
                for v in range(n):
                    if v != u:
                        if v in adj[u] and not visited[v]:
                            q.append(v)
                            visited[v] = True
                            reachable[v] = True
            
            for i in range(n):
                if not reachable[i]:
                    return False
            return True

        weights = sorted(list(set([w for _, _, w in edges])))
        
        left, right = 0, len(weights) - 1
        ans = -1
        while left <= right:
            mid = (left + right) // 2
            if check(weights[mid]):
                ans = weights[mid]
                right = mid - 1
            else:
                left = mid + 1

        return ans