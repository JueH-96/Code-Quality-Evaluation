class Solution:
    def maximumEnergy(self, energy: list[int], k: int) -> int:
        n = len(energy)
        dp = [0] * n
        max_energy = float('-inf')
        
        for i in range(n-1, -1, -1):
            if i + k >= n:
                dp[i] = energy[i]
            else:
                dp[i] = energy[i] + dp[i + k]
            max_energy = max(max_energy, dp[i])
        
        return max_energy