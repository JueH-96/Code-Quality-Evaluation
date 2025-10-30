import math
from functools import reduce
from typing import List

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        g = reduce(math.gcd, nums)
        if g == 1:
            return 1
        else:
            cnt = nums.count(g)
            return (cnt + 1) // 2