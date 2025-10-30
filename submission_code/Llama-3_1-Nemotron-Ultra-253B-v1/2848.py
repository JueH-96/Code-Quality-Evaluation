from typing import List

class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        n = len(nums)
        mod = 10**9 + 7
        compat = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and (nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0):
                    compat[i][j] = True
        
        dp = [[0] * n for _ in range(1 << n)]
        for i in range(n):
            dp[1 << i][i] = 1
        
        for mask in range(1 << n):
            for last in range(n):
                if not (mask & (1 << last)):
                    continue
                for j in range(n):
                    if not (mask & (1 << j)) and compat[last][j]:
                        new_mask = mask | (1 << j)
                        dp[new_mask][j] = (dp[new_mask][j] + dp[mask][last]) % mod
        
        full_mask = (1 << n) - 1
        return sum(dp[full_mask][j] for j in range(n)) % mod