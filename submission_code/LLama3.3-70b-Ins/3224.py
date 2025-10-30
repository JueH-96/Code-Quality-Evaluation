from typing import List

class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        res = 1
        for i in range(1, len(sick)):
            res = res * (sick[i] - sick[i-1] - 1) % MOD
        res = res * (sick[0] + 1) % MOD * (n - sick[-1]) % MOD
        return res