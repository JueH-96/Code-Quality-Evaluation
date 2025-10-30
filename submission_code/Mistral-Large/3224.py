from typing import List

class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7

        # Function to calculate factorial modulo MOD
        def factorial(x):
            result = 1
            for i in range(2, x + 1):
                result = (result * i) % MOD
            return result

        # Function to calculate nCr modulo MOD
        def nCr(n, r):
            if r > n:
                return 0
            numer = factorial(n)
            denom = (factorial(r) * factorial(n - r)) % MOD
            return (numer * pow(denom, MOD - 2, MOD)) % MOD

        # Calculate the gaps between sick children and the boundaries
        gaps = []
        prev = 0
        for s in sick:
            if s > prev + 1:
                gaps.append(s - prev - 1)
            prev = s
        if prev < n - 1:
            gaps.append(n - prev - 1)

        # Calculate the number of ways to infect the gaps
        result = 1
        for gap in gaps:
            result = (result * nCr(gap + 1, gap)) % MOD

        return result