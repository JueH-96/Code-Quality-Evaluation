from typing import List

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        total_sum = sum(nums)
        
        # dp[i] will be the number of ways to get sum i using the elements of nums
        dp = [0] * (total_sum + 1)
        dp[0] = 1  # There's one way to get sum 0: using the empty subset
        
        for num in nums:
            for j in range(total_sum, num - 1, -1):
                dp[j] = (dp[j] + dp[j - num]) % MOD
        
        result = 0
        for s in range(l, r + 1):
            if s <= total_sum:
                result = (result + dp[s]) % MOD
        
        return result