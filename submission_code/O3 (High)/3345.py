from typing import List

MOD = 10 ** 9 + 7

class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # pre–compute powers of two
        pow2 = [1] * (n + 1)
        for i in range(1, n + 1):
            pow2[i] = (pow2[i - 1] << 1) % MOD      # 2 ^ i  (mod MOD)
        
        # dp[s][c] = number of subsequences whose sum == s and size == c
        dp = [[0] * (n + 1) for _ in range(k + 1)]
        dp[0][0] = 1                                # empty subsequence
        
        for val in nums:
            # traverse backwards to avoid re-using the same element twice
            for s in range(k, val - 1, -1):
                for c in range(n, 0, -1):
                    dp[s][c] = (dp[s][c] + dp[s - val][c - 1]) % MOD
        
        # Each valid subsequence S of size c contributes 2^(n-c)
        ans = 0
        for c in range(1, n + 1):
            if dp[k][c]:
                ans = (ans + dp[k][c] * pow2[n - c]) % MOD
        
        return ans