from math import gcd
from functools import reduce
from typing import List

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        overall_gcd = reduce(gcd, nums)
        return 1 if overall_gcd == 1 else 2