from typing import List

class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        if n < 2:
            return []
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i : n+1 : i] = [False] * len(sieve[i*i : n+1 : i])
        result = []
        for x in range(2, (n // 2) + 1):
            y = n - x
            if sieve[x] and sieve[y]:
                result.append([x, y])
        return result