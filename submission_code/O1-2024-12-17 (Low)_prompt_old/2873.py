class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        # Edge case: if n < 3, it's impossible for two primes to sum to n
        if n < 3:
            return []
        
        # Sieve of Eratosthenes to identify primes up to n
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False  # 0 and 1 are not primes
        
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        
        # Find pairs x, y where x + y = n and both are prime
        result = []
        for x in range(2, (n // 2) + 1):
            if sieve[x] and sieve[n - x]:
                result.append([x, n - x])
        
        return result