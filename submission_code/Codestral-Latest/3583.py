from math import gcd
from itertools import combinations
from typing import List

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        # Calculate all possible GCD pairs
        gcd_pairs = [gcd(nums[i], nums[j]) for i, j in combinations(range(len(nums)), 2)]

        # Sort the GCD pairs
        gcd_pairs.sort()

        # Return the results for the given queries
        return [gcd_pairs[query] for query in queries]