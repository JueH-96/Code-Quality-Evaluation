from typing import List

def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        prime_pairs = []
        for x in range(2, n // 2 + 1):
            y = n - x
            if is_prime(x) and is_prime(y):
                prime_pairs.append([x, y])
        return prime_pairs