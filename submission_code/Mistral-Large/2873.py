from typing import List

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

        primes = []
        for i in range(2, n + 1):
            if is_prime(i):
                primes.append(i)

        prime_pairs = []
        for x in primes:
            if x > n // 2:
                break
            y = n - x
            if y >= x and y in primes:
                prime_pairs.append([x, y])

        return prime_pairs