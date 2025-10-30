class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        def is_prime(num):
            if num < 2:
                return False
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    return False
            return True

        pairs = []
        for i in range(1, n//2 + 1):
            if is_prime(i) and is_prime(n - i):
                pairs.append([i, n - i])
        return pairs