from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        odd_count = 0
        even_count = 0
        for num in nums:
            if num % 2 == 0:
                even_count += 1
            else:
                odd_count += 1
        if odd_count == 0 or even_count == 0:
            return len(nums)
        return max(odd_count, even_count)