from math import gcd
from bisect import insort

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        gcdPairs = []
        n = len(nums)
        
        for i in range(n):
            for j in range(i + 1, n):
                insort(gcdPairs, gcd(nums[i], nums[j]))
                
        return [gcdPairs[i] for i in queries]