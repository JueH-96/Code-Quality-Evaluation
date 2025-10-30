from typing import List

class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        def is_prime(num):
            if num <= 1:
                return False
            if num == 2:
                return True
            if num % 2 == 0:
                return False
            for i in range(3, int(num**0.5) + 1, 2):
                if num % i == 0:
                    return False
            return True
        
        primes = [i for i in range(2, n) if is_prime(i)]
        prime_set = set(primes)
        result = []
        
        for x in primes:
            y = n - x
            if y >= x and y in prime_set:
                result.append([x, y])
        
        return result