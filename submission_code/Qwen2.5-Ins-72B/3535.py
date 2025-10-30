from typing import List

class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        dp = [1] * 51
        count = [0] * 51
        
        for num in nums:
            for i in range(num + 1):
                dp[i] = (dp[i] + dp[num - i]) % MOD
            for i in range(num + 1):
                dp[i] = (dp[i] - dp[num - i - 1] if num - i - 1 >= 0 else dp[i]) % MOD
            count[num] += 1
        
        result = 1
        for num in nums:
            result = (result * dp[num]) % MOD
            for i in range(num + 1):
                dp[i] = (dp[i] - count[num]) % MOD
        
        return result