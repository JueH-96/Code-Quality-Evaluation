from math import gcd
from typing import List

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)
        gcd_pairs = []
        
        # Calculate GCD for all possible pairs and store them in gcd_pairs
        for i in range(n):
            for j in range(i + 1, n):
                gcd_pairs.append(gcd(nums[i], nums[j]))
        
        # Sort the gcd_pairs in ascending order
        gcd_pairs.sort()
        
        # Answer the queries
        answer = []
        for query in queries:
            answer.append(gcd_pairs[query])
        
        return answer