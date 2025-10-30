from math import gcd
from typing import List

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)
        gcd_pairs = []
        
        # Generate all possible gcd pairs and sort them
        for i in range(n):
            for j in range(i + 1, n):
                gcd_pairs.append(gcd(nums[i], nums[j]))
        gcd_pairs.sort()
        
        # Use a dictionary to store the index of each gcd value
        gcd_index = {}
        for i, val in enumerate(gcd_pairs):
            if val not in gcd_index:
                gcd_index[val] = i
        
        # Generate the result for each query
        result = []
        for query in queries:
            result.append(gcd_pairs[query])
        
        return result