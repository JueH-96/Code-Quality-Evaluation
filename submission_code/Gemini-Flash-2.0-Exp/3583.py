from typing import List
import math

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        gcd_pairs = []
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                gcd_pairs.append(math.gcd(nums[i], nums[j]))
        gcd_pairs.sort()
        
        answer = []
        for query in queries:
            answer.append(gcd_pairs[query])
        
        return answer