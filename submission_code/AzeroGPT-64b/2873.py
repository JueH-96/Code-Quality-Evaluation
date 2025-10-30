import math
from typing import List

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        # Generate prime numbers up to n
        primes = [i for i in range(2, n+1) if is_prime(i)]
        
        # Find prime pairs that sum to n
        prime_pairs = []
        for x in primes:
            if x > n - x: break  # Avoid duplicate pairs
            y = n - x
            if y in primes and y >= x:
                prime_pairs.append([x, y])
                
        return prime_pairs