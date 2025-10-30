from math import gcd
from functools import reduce

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        g = reduce(gcd, nums)
        min_element = min(nums)
        if min_element != g:
            return 1
        count = sum(1 for x in nums if x == min_element)
        return (count + 1) // 2