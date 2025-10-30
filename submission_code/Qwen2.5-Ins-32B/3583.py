from typing import List
from math import gcd
from itertools import combinations

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        gcdPairs = sorted(gcd(a, b) for a, b in combinations(nums, 2))
        return [gcdPairs[q] for q in queries]