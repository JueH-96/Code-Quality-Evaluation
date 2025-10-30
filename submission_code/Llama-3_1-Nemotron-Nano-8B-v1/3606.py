from typing import List

class Solution:
    def minElement(self, nums: List[int]) -> int:
        digit_sums = []
        for num in nums:
            digit_sum = sum(int(d) for d in str(num))
            digit_sums.append(digit_sum)
        return min(digit_sums)