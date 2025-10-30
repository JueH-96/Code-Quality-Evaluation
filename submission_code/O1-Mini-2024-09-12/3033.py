class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        mismatches = [i for i in range(n) if s1[i] != s2[i]]
        k = len(mismatches)
        
        if k % 2 != 0:
            return -1
        
        if k == 0:
            return 0
        
        # Initialize cost matrix
        INF = float('inf')
        cost = [[INF for _ in range(k)] for _ in range(k)]
        
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                pos_i = mismatches[i]
                pos_j = mismatches[j]
                if abs(pos_i - pos_j) == 1:
                    cost[i][j] = 1
                else:
                    cost[i][j] = x
        
        # Hungarian Algorithm for Minimum Weight Perfect Matching
        # Implementation based on https://pastebin.com/1vSWMEvQ
        def hungarian(matrix):
            n = len(matrix)
            m = len(matrix[0])
            u = [0] * (n+1)
            v = [0] * (m+1)
            p = [0] * (m+1)
            way = [0] * (m+1)
            for i in range(1, n+1):
                p[0] = i
                minv = [INF] * (m+1)
                used = [False] * (m+1)
                j0 = 0
                i0 = i
                delta = 0
                while True:
                    used[j0] = True
                    i0 = p[j0]
                    delta = INF
                    j1 = 0
                    for j in range(1, m+1):
                        if not used[j]:
                            cur = matrix[i0-1][j-1] - u[i0] - v[j]
                            if cur < minv[j]:
                                minv[j] = cur
                                way[j] = j0
                            if minv[j] < delta:
                                delta = minv[j]
                                j1 = j
                    if delta == INF:
                        break
                    for j in range(0, m+1):
                        if used[j]:
                            u[p[j]] += delta
                            v[j] -= delta
                        else:
                            minv[j] -= delta
                    j0 = j1
                    if p[j0] == 0:
                        break
                # augmenting path
                while True:
                    j1 = way[j0]
                    p[j0] = p[j1]
                    j0 = j1
                    if j0 == 0:
                        break
            # Total cost
            total_cost = 0
            for j in range(1, m+1):
                if p[j] != 0:
                    total_cost += matrix[p[j]-1][j-1]
            return total_cost
        
        total_cost = hungarian(cost)
        return total_cost if total_cost < INF else -1