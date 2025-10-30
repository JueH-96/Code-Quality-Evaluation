class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        dp = [[0] * n for _ in range(k)]
        for i in range(n):
            dp[0][i] = stayScore[0][i]
        for i in range(1, k):
            for j in range(n):
                for l in range(n):
                    if j == l:
                        dp[i][j] = max(dp[i][j], dp[i - 1][l] + stayScore[i][j])
                    else:
                        dp[i][j] = max(dp[i][j], dp[i - 1][l] + travelScore[l][j])
        return max(dp[k - 1])