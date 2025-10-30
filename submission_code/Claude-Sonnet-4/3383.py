class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        # dp[i] represents the maximum energy we can get starting from position i
        dp = [0] * n
        
        # Work backwards from the end
        for i in range(n - 1, -1, -1):
            if i + k >= n:
                # Can't jump further, so energy is just the current value
                dp[i] = energy[i]
            else:
                # Can jump to i+k, so take current energy plus best from i+k
                dp[i] = energy[i] + dp[i + k]
        
        # Return the maximum value among all possible starting positions
        return max(dp)