class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        # Special cases
        if n < 2:
            return []
        
        # Step 1: Generate a list of primes up to n using Sieve of Eratosthenes
        sieve = [True] * (n+1)
        sieve[0] = False
        sieve[1] = False
        p = 2
        while p * p <= n:
            if sieve[p]:
                for multiple in range(p*p, n+1, p):
                    sieve[multiple] = False
            p += 1
        
        # Step 2: For each prime x <= n//2, check if n-x is also prime
        result = []
        for x in range(2, (n // 2) + 1):
            if sieve[x] and sieve[n - x]:
                result.append([x, n - x])
        
        return result