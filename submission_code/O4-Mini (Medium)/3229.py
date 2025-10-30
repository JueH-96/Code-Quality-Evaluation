from bisect import bisect_left
from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        # Sort nums and build prefix sums
        A = sorted(nums)
        n = len(A)
        pref = [0] * (n + 1)
        for i in range(n):
            pref[i+1] = pref[i] + A[i]

        # Generate all palindromic numbers less than 10^9
        pals = []
        # odd-length palindromes: total length = 2*h - 1, for h = 1..5 (max length 9)
        for h in range(1, 6):
            start = 10**(h-1)
            end = 10**h
            for half in range(start, end):
                s = str(half)
                pal = int(s + s[-2::-1])  # mirror all but last digit
                if pal < 10**9:
                    pals.append(pal)
        # even-length palindromes: total length = 2*h, for h = 1..4 (max length 8)
        for h in range(1, 5):
            start = 10**(h-1)
            end = 10**h
            for half in range(start, end):
                s = str(half)
                pal = int(s + s[::-1])  # mirror entire half
                if pal < 10**9:
                    pals.append(pal)

        pals.sort()

        # Cost function using prefix sums: sum |A[i] - p|
        def cost(p: int) -> int:
            # idx = number of elements <= p
            idx = bisect_left(A, p)
            sum_left = pref[idx]
            sum_right = pref[n] - sum_left
            # cost = p*idx - sum_left + sum_right - p*(n-idx)
            return p*idx - sum_left + sum_right - p*(n-idx)

        # Find minimal cost over all palindromes
        ans = 10**30
        for p in pals:
            # Early prune: if cost difference lower bound exceeds current ans?
            # (Not strictly needed; full scan is fast enough.)
            c = cost(p)
            if c < ans:
                ans = c

        return ans