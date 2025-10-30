class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        # Edge case: No valid pairs if n < 2
        if n < 2:
            return []
        
        # Sieve of Eratosthenes to identify primes up to n
        prime = [True] * (n+1)
        prime[0] = False
        prime[1] = False
        
        p = 2
        while p*p <= n:
            if prime[p]:
                for multiple in range(p*p, n+1, p):
                    prime[multiple] = False
            p += 1
        
        # Find pairs (x, y) such that x + y = n and both x and y are prime
        result = []
        for x in range(2, (n // 2) + 1):
            if prime[x] and prime[n - x]:
                result.append([x, n - x])
                
        return result