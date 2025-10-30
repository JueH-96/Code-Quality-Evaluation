class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        dp = [[-float('inf')] * n for _ in range(4)]

        for j in range(n):
            dp[0][j] = a[0] * b[j]

        for k in range(1, 4):
            for j in range(k, n):
                max_prev_val = -float('inf')
                for i in range(k - 1, j):
                    max_prev_val = max(max_prev_val, dp[k - 1][i])
                if max_prev_val != -float('inf'):
                    dp[k][j] = max_prev_val + a[k] * b[j]

        max_score = -float('inf')
        for j in range(3, n):
            max_score = max(max_score, dp[3][j])

        return max_score