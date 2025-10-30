from typing import List
from math import gcd
from itertools import combinations

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        # Calculate all GCDs for pairs (i, j) where 0 <= i < j < n
        gcdPairs = [gcd(a, b) for a, b in combinations(nums, 2)]
        
        # Sort the gcdPairs in ascending order
        gcdPairs.sort()
        
        # Prepare the answer list based on the queries
        answer = [gcdPairs[q] for q in queries]
        
        return answer