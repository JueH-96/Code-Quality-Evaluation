class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        prefix_sum = [0] * (n + 1)
        dp = [[-inf] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0

        for i in range(1, n + 1):
            prefix_sum[i] = prefix_sum[i - 1] + energy[i - 1]
            for j in range(1, min(i, k) + 1):
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - 1] + energy[i - 1] + prefix_sum[i - j])

        return max(dp[n])