from math import gcd
from typing import List

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        g = 0
        for num in nums:
            g = gcd(g, num)

        if g == 1:
            return 1
        else:
            positive_count = sum(1 for num in nums if num > 0)
            if positive_count == 0:
                return 0  # Should not happen given positive integers constraint
            elif positive_count == 1:
                return 1
            else:
                return 1