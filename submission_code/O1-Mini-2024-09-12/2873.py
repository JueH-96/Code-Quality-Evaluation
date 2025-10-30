from typing import List
from math import isqrt

class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        if n < 2:
            return []
        
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        
        result = []
        for x in range(2, n // 2 + 1):
            if sieve[x] and sieve[n - x]:
                result.append([x, n - x])
        
        return result