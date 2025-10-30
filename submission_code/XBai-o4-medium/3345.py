from typing import List

class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # Precompute powers of 2 up to n
        pow2 = [1] * (n + 1)
        for i in range(1, n + 1):
            pow2[i] = (pow2[i-1] * 2) % MOD
        
        # Initialize DP table: dp[sum][count] = number of subsets
        dp = [[0] * (n + 1) for _ in range(k + 1)]
        dp[0][0] = 1
        
        for num in nums:
            # Iterate over sum in reverse to avoid overwriting issues
            for s in range(k, -1, -1):
                for m in range(n + 1):
                    if dp[s][m]:
                        new_s = s + num
                        if new_s <= k:
                            dp[new_s][m + 1] = (dp[new_s][m + 1] + dp[s][m]) % MOD
        
        # Calculate the final answer
        ans = 0
        for m in range(n + 1):
            ans = (ans + dp[k][m] * pow2[n - m]) % MOD
        
        return ans