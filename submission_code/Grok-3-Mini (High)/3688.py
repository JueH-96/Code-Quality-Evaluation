from typing import List
from collections import Counter

class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        n = len(nums)
        if all(num >= 0 for num in nums):
            return sum(nums)
        elif all(num < 0 for num in nums):
            return max(nums)
        else:
            def kadane_with_removal(x=None):
                current_sum = 0
                max_sum = float('-inf')
                for num in nums:
                    if x is None or num != x:
                        current_sum = max(num, current_sum + num)
                        max_sum = max(max_sum, current_sum)
                return max_sum
            
            ans = kadane_with_removal(None)  # No removal
            
            counter = Counter(nums)
            for x in counter:
                if x < 0 and counter[x] < n:
                    max_sum_x = kadane_with_removal(x)
                    ans = max(ans, max_sum_x)
            
            return ans