class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        def is_prime(num):
            if num <= 1:
                return False
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    return False
            return True

        prime_pairs = []
        for x in range(2, n // 2 + 1):
            y = n - x
            if is_prime(x) and is_prime(y):
                prime_pairs.append([x, y])
        if n % 2 == 0 and is_prime(n // 2):
            if is_prime(n // 2):
                prime_pairs.append([n // 2, n // 2])
        return prime_pairs