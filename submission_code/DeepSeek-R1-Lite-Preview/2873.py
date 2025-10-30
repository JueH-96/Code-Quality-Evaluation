class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        if n < 2:
            return []
        
        # Initialize sieve
        prime = [True] * (n + 1)
        prime[0], prime[1] = False, False
        
        # Sieve of Eratosthenes
        for i in range(2, int(math.isqrt(n)) + 1):
            if prime[i]:
                for multiple in range(i * i, n + 1, i):
                    prime[multiple] = False
        
        result = []
        for x in range(2, (n // 2) + 1):
            y = n - x
            if prime[x] and prime[y]:
                result.append([x, y])
        
        return result