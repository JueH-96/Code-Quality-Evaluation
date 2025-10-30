from typing import List

class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        if n < 4:
            return []
        
        # Sieve of Eratosthenes to determine primes up to n
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        p = 2
        while p * p <= n:
            if is_prime[p]:
                for multiple in range(p * p, n + 1, p):
                    is_prime[multiple] = False
            p += 1
        
        result = []
        # Iterate over primes from 2 to n//2 (inclusive)
        for x in range(2, n//2 + 1):
            y = n - x
            if x > y:
                continue
            if is_prime[x] and is_prime[y]:
                result.append([x, y])
                
        return result