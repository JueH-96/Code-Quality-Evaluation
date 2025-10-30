from typing import List

class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # dp[j][s] = number of ways to pick a subsequence of length j with sum s
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        
        # Build the dp table
        for x in nums:
            # iterate lengths and sums in reverse to avoid overwriting
            for j in range(n - 1, -1, -1):
                for s in range(k - x, -1, -1):
                    if dp[j][s]:
                        dp[j+1][s+x] = (dp[j+1][s+x] + dp[j][s]) % MOD
        
        # Precompute powers of 2
        pow2 = [1] * (n + 1)
        for i in range(1, n + 1):
            pow2[i] = (pow2[i-1] * 2) % MOD
        
        # Sum over all T with sum k, weighted by 2^(n - |T|)
        ans = 0
        for length in range(n + 1):
            cnt = dp[length][k]
            if cnt:
                ans = (ans + cnt * pow2[n - length]) % MOD
        
        return ans