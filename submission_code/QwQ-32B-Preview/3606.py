from typing import List

class Solution:
    def minElement(self, nums: List[int]) -> int:
        min_sum = float('inf')
        for num in nums:
            sum_digits = sum(int(d) for d in str(num))
            if sum_digits < min_sum:
                min_sum = sum_digits
        return min_sum