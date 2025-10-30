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
        for i in range(2, n):
            if is_prime(i):
                primes.append(i)
        
        prime_set = set(primes)
        result = []
        
        for x in primes:
            y = n - x
            if y >= x and y in prime_set:
                result.append([x, y])
        
        return result