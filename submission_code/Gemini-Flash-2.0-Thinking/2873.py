class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        is_prime = [True] * (n + 1)
        if n >= 0: is_prime[0] = False
        if n >= 1: is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for multiple in range(i * i, n + 1, i):
                    is_prime[multiple] = False

        prime_pairs = []
        for x in range(2, n // 2 + 1):
            y = n - x
            if is_prime[x] and is_prime[y]:
                prime_pairs.append([x, y])
        return prime_pairs