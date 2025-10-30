from typing import List

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        mod = 10**9 + 7
        inversions = [0]*n
        for end, cnt in requirements:
            inversions[end] += cnt
        prefix = [0]*(n+1)
        for i in range(n):
            prefix[i+1] = prefix[i] + inversions[i]
        result = 1
        for i in range(n, 0, -1):
            result = result * pow(i, prefix[i], mod) % mod
        return result