import math
from functools import reduce
from typing import List

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        g = reduce(math.gcd, nums)
        arr = [x // g for x in nums]
        c = arr.count(1)
        m = len(arr) - c
        
        if m == 0:
            return (c + 1) // 2
        else:
            if c == 0:
                return 1
            else:
                return (c + 1) // 2