from typing import List

class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        # Function to check if a number is prime
        def is_prime(num: int) -> bool:
            if num < 2:
                return False
            for i in range(2, int(num ** 0.5) + 1):
                if num % i == 0:
                    return False
            return True

        # Generate all prime numbers up to n
        primes = [i for i in range(2, n + 1) if is_prime(i)]

        # Find all prime pairs that sum up to n
        prime_pairs = [[x, n - x] for x in primes if n - x in primes and x <= n - x]

        return prime_pairs