from typing import List
from math import gcd
from collections import defaultdict

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)
        gcd_pairs = []
        
        # Calculate GCD for all pairs
        for i in range(n):
            for j in range(i + 1, n):
                gcd_pairs.append(gcd(nums[i], nums[j]))
        
        # Sort the GCD pairs
        gcd_pairs.sort()
        
        # Answer the queries
        answer = [gcd_pairs[query] for query in queries]
        
        return answer