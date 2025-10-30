import math
from typing import List

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        gcd_pairs = []
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                gcd_pairs.append(math.gcd(nums[i], nums[j]))
        gcd_pairs.sort()
        
        result = []
        for query in queries:
            result.append(gcd_pairs[query])
        
        return result