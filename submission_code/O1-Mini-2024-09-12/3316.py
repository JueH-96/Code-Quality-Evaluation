from typing import List
from math import comb

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        
        # Precompute combination values
        combs = [[0]*(n+1) for _ in range(n+1)]
        for i in range(n+1):
            combs[i][0] = 1
            for j in range(1, i+1):
                combs[i][j] = (combs[i-1][j-1] + combs[i-1][j]) % MOD
        
        result = 0
        for i in range(n):
            for j in range(i+1, n):
                diff = nums[j] - nums[i]
                # Number of elements between i and j
                between = j - i - 1
                # We need to choose k-2 elements from the remaining between elements
                if k-2 <= between:
                    result = (result + diff * combs[between][k-2]) % MOD
        return result