from typing import List
import math

class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        if not nums:
            return 0
        
        first = nums[0]
        if first % 2 == 0:
            even = first
            odd = -math.inf
        else:
            odd = first
            even = -math.inf
        
        for num in nums[1:]:
            if num % 2 == 0:
                # Current number is even
                option = max(even + num, odd + num - x)
                new_even = max(even, option)
                new_odd = odd
            else:
                # Current number is odd
                option = max(odd + num, even + num - x)
                new_odd = max(odd, option)
                new_even = even
            even, odd = new_even, new_odd
        
        return max(even, odd)