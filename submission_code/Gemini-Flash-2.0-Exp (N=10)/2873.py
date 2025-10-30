class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        def is_prime(num):
            if num <= 1:
                return False
            if num <= 3:
                return True
            if num % 2 == 0 or num % 3 == 0:
                return False
            i = 5
            while i * i <= num:
                if num % i == 0 or num % (i + 2) == 0:
                    return False
                i += 6
            return True

        result = []
        for x in range(2, n // 2 + 1):
            if is_prime(x):
                y = n - x
                if is_prime(y) and x <= y:
                    result.append([x, y])
        return result