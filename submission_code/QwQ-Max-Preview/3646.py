from typing import List
from collections import defaultdict

class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        sum_dict = defaultdict(int)
        count_dict = defaultdict(int)
        total_sum = 0
        
        for num in nums:
            # Calculate contributions from num-1 and num+1
            prev_lower_sum = sum_dict.get(num - 1, 0)
            prev_higher_sum = sum_dict.get(num + 1, 0)
            count_lower = count_dict.get(num - 1, 0)
            count_higher = count_dict.get(num + 1, 0)
            
            # Compute new sum and count for the current number
            new_sum = (num + prev_lower_sum + prev_higher_sum + num * (count_lower + count_higher)) % MOD
            new_count = (1 + count_lower + count_higher) % MOD
            
            # Update the dictionaries
            sum_dict[num] = new_sum
            count_dict[num] = new_count
            
            # Add to the total sum
            total_sum = (total_sum + new_sum) % MOD
        
        return total_sum