class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        dp = [float('-inf')] * n
        
        for i in range(n - 1, -1, -1):
            if i + k >= n:
                dp[i] = energy[i]
            else:
                dp[i] = energy[i] + dp[i + k]
        
        return max(dp)