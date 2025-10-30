from typing import List

class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        if nums[0] % 2 == 0:
            even = nums[0]
            odd = float('-inf')
        else:
            odd = nums[0]
            even = float('-inf')
        
        for num in nums[1:]:
            if num % 2 == 0:
                new_even_candidate = max(even + num, odd + num - x)
                new_even = max(even, new_even_candidate)
                new_odd = odd
            else:
                new_odd_candidate = max(odd + num, even + num - x)
                new_odd = max(odd, new_odd_candidate)
                new_even = even
            even, odd = new_even, new_odd
        
        return max(even, odd)