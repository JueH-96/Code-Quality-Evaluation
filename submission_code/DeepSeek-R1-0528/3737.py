class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        m = n // 2
        INF = 10**18
        dp = [[INF] * 3 for _ in range(3)]
        
        for a in range(3):
            for b in range(3):
                if a != b:
                    dp[a][b] = cost[0][a] + cost[n-1][b]
        
        for p in range(1, m):
            new_dp = [[INF] * 3 for _ in range(3)]
            for a_prev in range(3):
                for b_prev in range(3):
                    if dp[a_prev][b_prev] == INF:
                        continue
                    for c in range(3):
                        for d in range(3):
                            if c == d:
                                continue
                            if c == a_prev or d == b_prev:
                                continue
                            total = dp[a_prev][b_prev] + cost[p][c] + cost[n-1-p][d]
                            if total < new_dp[c][d]:
                                new_dp[c][d] = total
            dp = new_dp
        
        ans = INF
        for i in range(3):
            for j in range(3):
                if dp[i][j] < ans:
                    ans = dp[i][j]
        return ans