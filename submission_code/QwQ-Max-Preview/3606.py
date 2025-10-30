from typing import List

class Solution:
    def minElement(self, nums: List[int]) -> int:
        digit_sums = [sum(int(digit) for digit in str(num)) for num in nums]
        return min(digit_sums)