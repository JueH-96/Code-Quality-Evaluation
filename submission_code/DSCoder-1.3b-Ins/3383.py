class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        dp = [0] * n
        dp[0] = energy[0]
        for i in range(1, n):
            dp[i] = max(dp[i-1] + energy[i], energy[i])
            if i >= k:
                dp[i] = max(dp[i], dp[i-k] + energy[i])
        return max(dp)