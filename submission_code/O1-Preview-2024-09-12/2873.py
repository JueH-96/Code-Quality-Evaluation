class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        sieve = [True] * (n + 1)
        sieve[0], sieve[1] = False, False
        for i in range(2, int(n ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        result = []
        for x in range(2, n // 2 + 1):
            if sieve[x]:
                y = n - x
                if y >= x and sieve[y]:
                    result.append([x, y])
        return result