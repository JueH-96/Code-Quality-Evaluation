from typing import List

class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        if n < 2:
            return []
        
        # Sieve of Eratosthenes to find all primes up to n
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        
        primes = [i for i in range(n + 1) if is_prime[i]]
        prime_pairs = []
        
        # Find pairs (x, y) such that x + y = n and both are prime
        prime_set = set(primes)
        for x in primes:
            y = n - x
            if y >= x and y in prime_set:
                prime_pairs.append([x, y])
        
        return prime_pairs