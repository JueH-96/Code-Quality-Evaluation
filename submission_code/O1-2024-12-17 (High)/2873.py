class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        # Edge case: if n < 2, there can be no prime pairs
        if n < 2:
            return []
        
        # Sieve of Eratosthenes to identify primes up to n
        is_prime = [True] * (n + 1)
        is_prime[0] = False
        is_prime[1] = False
        
        # Mark non-primes using the sieve
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
                    
        # Find all prime pairs (x, y) such that x + y == n and x <= y
        pairs = []
        for x in range(2, (n // 2) + 1):
            y = n - x
            if y < x:
                break
            if is_prime[x] and is_prime[y]:
                pairs.append([x, y])
        
        return pairs