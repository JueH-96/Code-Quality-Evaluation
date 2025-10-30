from typing import List

class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        dp = [[0] * (1 << n) for _ in range(n)]
        
        for mask in range(1, 1 << n):
            for i in range(n):
                if mask & (1 << i):
                    if bin(mask).count('1') == 1:
                        dp[i][mask] = 1
                    else:
                        for j in range(n):
                            if i != j and mask & (1 << j) and (nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0):
                                dp[i][mask] += dp[j][mask ^ (1 << i)]
                                dp[i][mask] %= MOD
        
        return sum(dp[i][(1 << n) - 1] for i in range(n)) % MOD