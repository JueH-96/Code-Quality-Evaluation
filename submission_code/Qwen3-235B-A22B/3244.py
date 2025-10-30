from math import gcd
from typing import List

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        m = min(nums)
        g = 0
        for num in nums:
            g = gcd(g, num)
        if g < m:
            return 1
        count = 0
        for num in nums:
            if num == m:
                count += 1
        return (count + 1) // 2