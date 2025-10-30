class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        if n < 2:
            return []
        # Create a sieve of Eratosthenes up to n
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if sieve[i]:
                sieve[i*i : n+1 : i] = [False] * len(sieve[i*i : n+1 : i])
        result = []
        # Check each x from 2 up to n//2
        for x in range(2, (n // 2) + 1):
            if sieve[x] and sieve[n - x]:
                result.append([x, n - x])
        return result