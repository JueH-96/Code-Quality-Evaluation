from math import factorial
from typing import List

class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        sick = [0] + sick + [n-1]
        diff = [sick[i+1] - sick[i] - 1 for i in range(len(sick)-1)]
        total = sum(diff)
        
        res = factorial(total)
        for d in diff:
            if d > 0:
                res = res // factorial(d)
        for i in range(1, len(diff) - 1):
            if diff[i] == 0:
                res = res * pow(2, diff[i-1], MOD) % MOD
        return res % MOD