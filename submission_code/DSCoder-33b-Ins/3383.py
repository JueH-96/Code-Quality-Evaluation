class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        dp = [0] * n
        dp[0] = energy[0]
        max_energy = dp[0]

        for i in range(1, n):
            dp[i] = energy[i]
            for j in range(1, min(i, k) + 1):
                dp[i] = max(dp[i], dp[i - j] + energy[i])
            max_energy = max(max_energy, dp[i])

        return max_energy