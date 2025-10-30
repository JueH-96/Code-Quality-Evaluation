from collections import Counter

class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        max_val = max(nums)
        dp = [Counter() for _ in range(max_val + 1)]
        dp[0][0, 0] = 1
        
        for x in range(1, max_val + 1):
            for l in range(x + 1):
                for r in range(max_val - x + 1):
                    dp[x][(l, r)] = (dp[x - 1][(l - 1, r)] if l > 0 else 0) + (dp[x - 1][(l, r - 1)] if r > 0 else 0)
        
        ans = 0
        for x in nums:
            for l in range(x + 1):
                for r in range(max_val - x + 1):
                    ans += dp[x][(l, r)]
        
        return ans % MOD