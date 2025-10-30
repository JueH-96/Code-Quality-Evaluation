class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        def is_prime(num):
            if num <= 1:
                return False
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    return False
            return True

        result = []
        for x in range(2, n // 2 + 1):
            y = n - x
            if x <= y and is_prime(x) and is_prime(y):
                result.append([x, y])
        return result