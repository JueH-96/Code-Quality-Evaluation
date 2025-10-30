import math
from functools import reduce

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        g = reduce(math.gcd, nums)
        count_g = sum(1 for x in nums if x == g)
        k = count_g if count_g > 0 else 1
        return (k + 1) // 2