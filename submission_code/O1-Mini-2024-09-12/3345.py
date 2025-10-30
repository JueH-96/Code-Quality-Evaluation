from typing import List

class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # Initialize DP table where dp[l][j] represents the number of subsequences
        # of length l with sum j
        dp = [ [0] * (k+1) for _ in range(n+1)]
        dp[0][0] = 1  # Base case: one subsequence of length 0 with sum 0
        
        for num in nums:
            for l in range(n, 0, -1):
                for j in range(k, num-1, -1):
                    dp[l][j] = (dp[l][j] + dp[l-1][j-num]) % MOD
        
        # Precompute powers of 2
        pow2 = [1] * (n+1)
        for i in range(1, n+1):
            pow2[i] = pow2[i-1] * 2 % MOD
        
        # Calculate the final result by summing over all possible lengths
        result = 0
        for l in range(1, n+1):
            result = (result + dp[l][k] * pow2[n - l]) % MOD
        
        return result