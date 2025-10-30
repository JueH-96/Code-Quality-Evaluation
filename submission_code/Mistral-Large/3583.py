from typing import List
import math
from bisect import bisect_left

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)
        gcd_pairs = []

        # Calculate GCD for all pairs (nums[i], nums[j]) where i < j
        for i in range(n):
            for j in range(i + 1, n):
                gcd_pairs.append(math.gcd(nums[i], nums[j]))

        # Sort the GCD pairs
        gcd_pairs.sort()

        # Answer the queries
        result = []
        for query in queries:
            result.append(gcd_pairs[query])

        return result