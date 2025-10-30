from typing import List
from collections import defaultdict

class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        count = defaultdict(int)  # count[x] is the number of subsequences ending with x
        sums = defaultdict(int)   # sums[x] is the sum of all subsequences ending with x
        total = 0  # To store the total sum of all good subsequences
        
        for x in nums:
            left_count = count[x - 1]
            right_count = count[x + 1]
            
            # Calculate new_count and new_sum for the current value x
            new_count = (1 + left_count + right_count) % MOD
            
            left_sum_part = (sums[x - 1] + (x * left_count) % MOD) % MOD
            right_sum_part = (sums[x + 1] + (x * right_count) % MOD) % MOD
            new_sum = (x + left_sum_part + right_sum_part) % MOD
            
            # Update the total sum
            total = (total + new_sum) % MOD
            
            # Update count and sums for current x
            count[x] = (count[x] + new_count) % MOD
            sums[x] = (sums[x] + new_sum) % MOD
        
        return total