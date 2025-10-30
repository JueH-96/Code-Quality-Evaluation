class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        dp = [0] * n
        dp[0] = energy[0]
        max_energy = energy[0]
        
        for i in range(1, n):
            dp[i] = energy[i]
            if (i + k) < n:
                dp[i] += dp[i - k]
            max_energy = max(max_energy, dp[i])
        
        return max_energy