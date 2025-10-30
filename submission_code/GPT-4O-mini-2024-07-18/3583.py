from typing import List
from math import gcd
from itertools import combinations

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        # Generate all GCD pairs
        gcdPairs = [gcd(a, b) for a, b in combinations(nums, 2)]
        # Sort the GCD pairs
        gcdPairs.sort()
        # Prepare the answer based on queries
        answer = [gcdPairs[q] for q in queries]
        return answer