from typing import List
import math

class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        even = -math.inf
        odd = -math.inf
        
        if nums[0] % 2 == 0:
            even = nums[0]
        else:
            odd = nums[0]
        
        for num in nums[1:]:
            if num % 2 == 0:
                candidate = max(even + num, odd + num - x)
                new_even = max(even, candidate)
                new_odd = odd
            else:
                candidate = max(odd + num, even + num - x)
                new_odd = max(odd, candidate)
                new_even = even
            
            even, odd = new_even, new_odd
        
        return max(even, odd)