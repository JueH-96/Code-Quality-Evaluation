from typing import List
import math

class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        if n < 4:
            return []
        
        # Sieve of Eratosthenes to generate primes up to n.
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.sqrt(n)) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        
        result = []
        # iterate through possible x
        for x in range(2, n // 2 + 1):
            y = n - x
            if y < x:
                continue
            if sieve[x] and sieve[y]:
                result.append([x, y])
        return result