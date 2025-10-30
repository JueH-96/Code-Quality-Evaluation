from typing import List

class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + (1 if possible[i] == 1 else -1)
        total = prefix[n]
        for k in range(1, n):
            if 2 * prefix[k] > total:
                return k
        return -1