class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        if n < 4:  # minimum sum of two primes is 2+2=4
            return []
        
        # Sieve of Eratosthenes to find all primes up to n
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        
        result = []
        
        # Check each potential x value
        for x in range(2, n // 2 + 1):
            if is_prime[x]:
                y = n - x
                if is_prime[y] and x <= y:
                    result.append([x, y])
        
        return result