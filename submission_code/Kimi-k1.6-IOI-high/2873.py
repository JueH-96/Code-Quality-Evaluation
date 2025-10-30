class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        if n < 2:
            return []
        
        # Create the sieve of Eratosthenes up to n
        sieve = [True] * (n + 1)
        sieve[0], sieve[1] = False, False
        for i in range(2, int(n ** 0.5) + 1):
            if sieve[i]:
                sieve[i*i : n+1 : i] = [False] * len(sieve[i*i : n+1 : i])
        
        result = []
        # Check all x from 2 to n//2 inclusive
        for x in range(2, (n // 2) + 1):
            y = n - x
            if sieve[x] and sieve[y]:
                result.append([x, y])
        
        return result