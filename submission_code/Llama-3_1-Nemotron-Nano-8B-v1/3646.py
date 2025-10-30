from collections import defaultdict
from typing import List

class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        dp = defaultdict(lambda: (0, 0))  # (sum, count)
        
        for num in nums:
            x = num
            # Get contributions from x-1 and x+1
            sum_prev, cnt_prev = dp[x - 1]
            sum_next, cnt_next = dp[x + 1]
            
            new_sum = x
            new_count = 1
            
            # Add contributions from x-1
            new_sum += sum_prev + cnt_prev * x
            new_count += cnt_prev
            
            # Add contributions from x+1
            new_sum += sum_next + cnt_next * x
            new_count += cnt_next
            
            # Update the current x in dp with modulo
            existing_sum, existing_count = dp[x]
            dp[x] = (
                (existing_sum + new_sum) % MOD,
                (existing_count + new_count) % MOD
            )
        
        # Sum all the sum values in dp and apply modulo
        total = 0
        for s, _ in dp.values():
            total = (total + s) % MOD
        return total