from math import gcd
from typing import List

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        # Calculate the GCD of all possible pairs
        gcd_pairs = []
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                gcd_pairs.append(gcd(nums[i], nums[j]))
        
        # Sort the GCD pairs in ascending order
        gcd_pairs.sort()
        
        # Find the element at index queries[i] in gcdPairs
        answer = []
        for query in queries:
            answer.append(gcd_pairs[query])
        
        return answer