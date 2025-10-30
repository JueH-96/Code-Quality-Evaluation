class Solution:
    def maximumEnergy(self, energy, k):
        n = len(energy)
        dp = [0] * n
        dp[0] = energy[0]
        for i in range(1, n):
            dp[i] = max(dp[i-1], energy[i] + (dp[i-k] if i >= k else 0))
        return max(dp)