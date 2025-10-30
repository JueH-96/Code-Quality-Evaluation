from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        even_count = 0
        odd_count = 0
        for num in nums:
            if num % 2 == 0:
                even_count += 1
            else:
                odd_count += 1
        case1 = max(even_count, odd_count)
        
        even_len = 0
        odd_len = 0
        for num in nums:
            if num % 2 == 0:
                new_even = max(even_len, (odd_len + 1) if odd_len > 0 else 1)
                new_odd = odd_len
            else:
                new_odd = max(odd_len, (even_len + 1) if even_len > 0 else 1)
                new_even = even_len
            even_len, odd_len = new_even, new_odd
        case2 = max(even_len, odd_len)
        
        return max(case1, case2)