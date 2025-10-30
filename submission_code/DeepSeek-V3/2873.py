class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        if n < 2:
            return []
        
        # Sieve of Eratosthenes to find all primes up to n
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        
        primes = [i for i, is_prime in enumerate(sieve) if is_prime]
        
        result = []
        for x in primes:
            if x > n // 2:
                break
            y = n - x
            if sieve[y]:
                result.append([x, y])
        
        return result