class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        dp = [0]*n
        
        # Build dp array from the end to the beginning
        for i in range(n-1, -1, -1):
            dp[i] = energy[i]
            if i + k < n:
                dp[i] += dp[i + k]
        
        # The answer is the maximum value in dp
        return max(dp)