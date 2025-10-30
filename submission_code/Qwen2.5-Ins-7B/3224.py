class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        gaps = [sick[i+1] - sick[i] - 1 for i in range(len(sick) - 1)]
        gaps = [0] + gaps + [n - 1 - sick[-1]]
        from math import factorial
        from functools import reduce
        from operator import mul
        
        def inv(x):
            return pow(x, MOD-2, MOD)
        
        def ncr(n, r):
            return reduce(mul, (inv(factorial(i)) * factorial(n-r+i) for i in range(1, r+1)), 1) % MOD
        
        total = 1
        for gap in gaps:
            total = (total * ncr(gap + 1, 1)) % MOD
        return total