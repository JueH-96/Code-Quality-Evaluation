from typing import List
import math

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        gcdPairs = []
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                gcdPairs.append(math.gcd(nums[i], nums[j]))
        gcdPairs.sort()
        
        result = []
        for q in queries:
            result.append(gcdPairs[q])
        return result