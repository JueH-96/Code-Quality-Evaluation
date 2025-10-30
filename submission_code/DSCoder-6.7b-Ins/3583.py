from typing import List
from math import gcd
from itertools import combinations

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        gcd_pairs = sorted([gcd(i, j) for i, j in combinations(nums, 2)])
        return [gcd_pairs[i] for i in queries]