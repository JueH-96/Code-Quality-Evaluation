class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        dp = [float('-inf')] * n
        dp[0] = energy[0]
        
        for i in range(1, n):
            dp[i] = max(dp[i], dp[(i - k + n) % n] + energy[i])
        
        return max(dp)