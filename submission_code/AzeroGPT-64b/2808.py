class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        dp = [[10000000 for _ in range(n+1)] for _ in range(n+1)]
        dp[0][0] = 0

        for i in range(1, n+1):
            dp[i][0] = dp[i-1][0]
            dp[i][i] = min(dp[i-1][i], cost[i-1])
            for j in range(1, i):
                dp[i][j] = dp[i-1][j]
                if j > time[i-1]:
                    dp[i][j] = min(dp[i][j], dp[i-1][j-time[i-1]-1] + cost[i-1])
            if i < n:
                dp[i][i+1] = min(dp[i-1][i+1], cost[i-1])
                for j in range(i+2, n+1):
                    if j > time[i-1]:
                        dp[i][j] = min(dp[i-1][j], dp[i][j], dp[i-1][j-time[i-1]-1] + cost[i-1])  
        l, r = 0, n
        while l < r:
            mid = (l + r) // 2
            if dp[n][mid+1] < 10000000:
                r = mid
            else:
                l = mid + 1
        return dp[n][l]