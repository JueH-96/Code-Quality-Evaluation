from typing import List
from collections import defaultdict

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        counters = [defaultdict(int) for _ in range(n)]
        for x, y in pick:
            counters[x][y] += 1
        result = 0
        for i in range(n):
            for cnt in counters[i].values():
                if cnt > i:
                    result += 1
                    break
        return result