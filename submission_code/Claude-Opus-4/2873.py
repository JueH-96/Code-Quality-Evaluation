class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        # Handle edge cases
        if n < 4:
            return []
        
        # Sieve of Eratosthenes to find all primes up to n
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        
        # Find all prime pairs
        result = []
        
        # Only need to check up to n/2 to avoid duplicates
        for x in range(2, n // 2 + 1):
            if is_prime[x]:
                y = n - x
                if is_prime[y]:
                    result.append([x, y])
        
        return result