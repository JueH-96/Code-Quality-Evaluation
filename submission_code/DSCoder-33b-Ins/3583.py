from typing import List
from math import gcd
from sortedcontainers import SortedList

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)
        gcd_pairs = SortedList()
        for i in range(n):
            for j in range(i + 1, n):
                gcd_pairs.add(gcd(nums[i], nums[j]))
        return [gcd_pairs[i] for i in queries]