class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        if n < 2:
            return []
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for num in range(2, int(n ** 0.5) + 1):
            if sieve[num]:
                sieve[num*num : n+1 : num] = [False] * len(sieve[num*num : n+1 : num])
        result = []
        for x in range(2, (n // 2) + 1):
            y = n - x
            if sieve[x] and sieve[y]:
                result.append([x, y])
        return result