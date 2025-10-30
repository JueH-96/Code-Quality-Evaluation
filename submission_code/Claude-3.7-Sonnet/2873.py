class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        if n < 4:  # No prime pair can sum to less than 4 (since smallest prime is 2)
            return []
        
        # Use Sieve of Eratosthenes to identify all primes up to n
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        
        p = 2
        while p * p <= n:
            if is_prime[p]:
                for i in range(p * p, n + 1, p):
                    is_prime[i] = False
            p += 1
        
        # Find prime pairs that sum to n
        result = []
        for x in range(2, n // 2 + 1):  # We only need to check up to n/2
            y = n - x
            if is_prime[x] and is_prime[y]:
                result.append([x, y])
        
        return result