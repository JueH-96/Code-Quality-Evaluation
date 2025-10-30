class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        if n < 4:
            return [] if n < 4 or n == 3 else [[2, 2]]  # Handle small n cases directly for optimization
        
        # Create a boolean array "is_prime[0..n]" and initialize all entries as true.
        is_prime = [True] * (n + 1)
        is_prime[0] = False
        is_prime[1] = False
        
        # Sieve of Eratosthenes to find all primes up to n
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    if j != i:  # Ensure we don't mark the prime itself as non-prime
                        is_prime[j] = False
        
        # Find all pairs (x, y) such that x + y = n and both are prime
        result = []
        for x in range(2, n // 2 + 1):
            if is_prime[x] and is_prime[n - x]:
                result.append([x, n - x])
        
        return result