class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        m = n // 2
        INF = float('inf')
        dp = [[INF] * 3 for _ in range(3)]
        
        for a in range(3):
            for b in range(3):
                if a != b:
                    dp[a][b] = cost[m-1][a] + cost[n-1 - (m-1)][b]
        
        for k in range(m-2, -1, -1):
            new_dp = [[INF] * 3 for _ in range(3)]
            for a in range(3):
                for b in range(3):
                    if a == b:
                        continue
                    total = cost[k][a] + cost[n-1-k][b]
                    best_next = INF
                    for c in range(3):
                        for d in range(3):
                            if c == d:
                                continue
                            if c == a or d == b:
                                continue
                            if dp[c][d] < best_next:
                                best_next = dp[c][d]
                    if best_next != INF:
                        new_dp[a][b] = total + best_next
            dp = new_dp
        
        ans = INF
        for a in range(3):
            for b in range(3):
                if a != b and dp[a][b] < ans:
                    ans = dp[a][b]
        return ans