class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        k = len(a)

        dp = [[float('-inf')] * n for _ in range(k)]

        # Base case: selecting the first element
        for i in range(n):
            dp[0][i] = a[0] * b[i]

        for i_a in range(1, k):
            for i_b in range(i_a, n):
                max_prev_score = float('-inf')
                for prev_index in range(i_a):
                    if dp[i_a - 1][prev_index] != float('-inf'):
                        max_prev_score = max(max_prev_score, dp[i_a - 1][prev_index])

                if max_prev_score != float('-inf'):
                    dp[i_a][i_b] = max_prev_score + a[i_a] * b[i_b]

                # Optimization:
                max_prev_dp = float('-inf')
                for j in range(i_b):
                    max_prev_dp = max(max_prev_dp, dp[i_a - 1][j])

                if max_prev_dp != float('-inf'):
                    dp[i_a][i_b] = max(dp[i_a][i_b], max_prev_dp + a[i_a] * b[i_b])

        return max(dp[k - 1]) if any(val != float('-inf') for val in dp[k-1]) else 0