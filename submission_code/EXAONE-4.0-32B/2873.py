class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        if n < 2:
            return []
        
        is_prime = [True] * (n + 1)
        is_prime[0] = False
        is_prime[1] = False
        p = 2
        while p * p <= n:
            if is_prime[p]:
                for i in range(p * p, n + 1, p):
                    is_prime[i] = False
            p += 1
        
        res = []
        for x in range(2, n // 2 + 1):
            y = n - x
            if is_prime[x] and is_prime[y]:
                res.append([x, y])
        return res