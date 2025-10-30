class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)
        P = [0] * (n + 1)
        for i in range(1, n + 1):
            P[i] = P[i - 1] + nums[i - 1]
        
        NEG_INF = -10**18
        dp = [[NEG_INF] * (k + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = 0
        
        for j in range(1, k + 1):
            best = NEG_INF
            for i in range(n + 1):
                if i >= m:
                    candidate_best = dp[i - m][j - 1] - P[i - m]
                    if candidate_best > best:
                        best = candidate_best
                if i > 0:
                    dp[i][j] = dp[i - 1][j]
                if i >= m:
                    total = best + P[i]
                    if total > dp[i][j]:
                        dp[i][j] = total
        return dp[n][k]