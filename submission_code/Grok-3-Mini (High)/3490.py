from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        count_even = 0
        count_odd = 0
        max_end_even = 0
        max_end_odd = 0
        alt_max = 0
        
        for num in nums:
            p = num % 2
            if p == 0:
                count_even += 1
            else:
                count_odd += 1
            
            if p == 0:
                opp_max_val = max_end_odd
            else:
                opp_max_val = max_end_even
            
            if opp_max_val > 0:
                len_end = 1 + opp_max_val
            else:
                len_end = 1
            
            if p == 0:
                max_end_even = max(max_end_even, len_end)
            else:
                max_end_odd = max(max_end_odd, len_end)
            
            alt_max = max(alt_max, len_end)
        
        return max(count_even, count_odd, alt_max)