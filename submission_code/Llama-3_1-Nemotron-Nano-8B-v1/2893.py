from typing import List

class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        even = float('-inf')
        odd = float('-inf')
        first_p = nums[0] % 2
        if first_p == 0:
            even = nums[0]
        else:
            odd = nums[0]
        
        for i in range(1, len(nums)):
            current = nums[i]
            current_p = current % 2
            if current_p == 0:
                new_even = max(even, even + current, odd + current - x)
                new_odd = odd
            else:
                new_odd = max(odd, even + current - x, odd + current)
                new_even = even
            even, odd = new_even, new_odd
        
        return max(even, odd)