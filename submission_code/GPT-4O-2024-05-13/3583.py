from typing import List
from math import gcd
from itertools import combinations

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        gcdPairs = []
        
        # Generate all possible pairs and calculate their GCD
        for i, j in combinations(range(len(nums)), 2):
            gcdPairs.append(gcd(nums[i], nums[j]))
        
        # Sort the gcdPairs in ascending order
        gcdPairs.sort()
        
        # Retrieve the values based on the queries
        answer = [gcdPairs[q] for q in queries]
        
        return answer