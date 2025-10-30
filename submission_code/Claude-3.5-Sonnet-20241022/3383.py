class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        dp = [-float('inf')] * n
        
        # Base cases - last k positions
        for i in range(n-1, n-k-1, -1):
            dp[i] = energy[i]
            
        # Fill dp array from right to left
        for i in range(n-k-1, -1, -1):
            dp[i] = energy[i] + dp[i+k]
            
        # Return maximum value possible starting from any position
        return max(dp)