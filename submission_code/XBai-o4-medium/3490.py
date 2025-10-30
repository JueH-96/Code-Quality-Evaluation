from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        even_count = 0
        odd_count = 0
        even_len = 0
        odd_len = 0
        for num in nums:
            if num % 2 == 0:
                even_count += 1
                new_even = max(even_len, odd_len + 1)
                even_len = new_even
            else:
                odd_count += 1
                new_odd = max(odd_len, even_len + 1)
                odd_len = new_odd
        option1 = max(even_count, odd_count)
        option2 = max(even_len, odd_len)
        return max(option1, option2)