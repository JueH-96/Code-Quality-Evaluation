from typing import List
import math

class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        """
        Return all pairs [x, y] such that:
            1 <= x <= y <= n
            x + y == n
            x and y are prime numbers
        The pairs are returned sorted by increasing x.
        """
        # For any n smaller than 4 there cannot be a valid pair,
        # because the smallest possible prime pair is (2, 2) with sum 4.
        if n < 4:
            return []

        # -------- 1. Build a prime sieve up to n --------
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        limit = int(math.isqrt(n))
        for p in range(2, limit + 1):
            if sieve[p]:
                sieve[p * p : n + 1 : p] = [False] * (((n - p * p) // p) + 1)

        # -------- 2. Collect the prime pairs --------
        res: List[List[int]] = []
        # Ensure x <= y => x <= n // 2 is sufficient
        for x in range(2, n // 2 + 1):
            if sieve[x]:
                y = n - x
                if y >= x and sieve[y]:
                    res.append([x, y])

        return res