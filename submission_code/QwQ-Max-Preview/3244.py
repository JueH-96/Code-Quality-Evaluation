from math import gcd
from functools import reduce
from typing import List

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        d = reduce(gcd, nums)
        if d == 1:
            return 1
        count = sum(1 for num in nums if num == d)
        return (count + 1) // 2