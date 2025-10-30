import math
from typing import List

class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 1000000007
        MAX_NUM = 100000
        n = len(nums)
        
        # Compute dp_end: number of good subsequences ending at each index
        sum_dp_val = [0] * (MAX_NUM + 1)
        dp_end = [0] * n
        for i in range(n):
            val = nums[i]
            dp_end_i = 1
            if val - 1 >= 0:
                dp_end_i += sum_dp_val[val - 1]
                dp_end_i %= MOD
            if val + 1 <= MAX_NUM:
                dp_end_i += sum_dp_val[val + 1]
                dp_end_i %= MOD
            dp_end[i] = dp_end_i
            sum_dp_val[val] = (sum_dp_val[val] + dp_end_i) % MOD
        
        # Compute dp_start: number of good subsequences starting at each index
        sum_after = [0] * (MAX_NUM + 1)
        dp_start = [0] * n
        for i in range(n - 1, -1, -1):
            val = nums[i]
            dp_start_i = 1
            if val - 1 >= 0:
                dp_start_i += sum_after[val - 1]
                dp_start_i %= MOD
            if val + 1 <= MAX_NUM:
                dp_start_i += sum_after[val + 1]
                dp_start_i %= MOD
            dp_start[i] = dp_start_i
            sum_after[val] = (sum_after[val] + dp_start_i) % MOD
        
        # Compute the sum of all good subsequences
        ans = 0
        for i in range(n):
            prod = (dp_end[i] * dp_start[i]) % MOD
            contrib = (nums[i] * prod) % MOD
            ans = (ans + contrib) % MOD
        
        return ans