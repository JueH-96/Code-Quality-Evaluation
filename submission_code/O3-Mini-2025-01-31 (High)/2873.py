from typing import List
import math

class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        # For n less than 2, no prime pairs are possible.
        if n < 2:
            return []
        
        # Create a sieve for prime numbers up to n using the Sieve of Eratosthenes.
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(math.sqrt(n)) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        
        # Find all pairs [x, y] such that:
        #    1 <= x <= y <= n, 
        #    x + y == n, and 
        #    both x and y are prime numbers.
        pairs = []
        # Iterate x from 2 to n//2 because x must be <= n/2 to keep x <= y.
        for x in range(2, n // 2 + 1):
            y = n - x
            if sieve[x] and sieve[y]:
                pairs.append([x, y])
        
        return pairs