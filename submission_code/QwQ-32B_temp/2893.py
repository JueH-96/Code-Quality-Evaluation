from typing import List

class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        first = nums[0]
        if first % 2 == 0:
            even_max = first
            odd_max = -float('inf')
        else:
            odd_max = first
            even_max = -float('inf')
        
        for num in nums[1:]:
            if num % 2 == 0:
                # Current number is even
                temp_even = max(even_max + num, odd_max + num - x)
                new_even = max(even_max, temp_even)
                new_odd = odd_max  # no change since current is even
                even_max, odd_max = new_even, new_odd
            else:
                # Current number is odd
                temp_odd = max(odd_max + num, even_max + num - x)
                new_odd = max(odd_max, temp_odd)
                new_even = even_max  # no change since current is odd
                even_max, odd_max = new_even, new_odd
        
        return max(even_max, odd_max)