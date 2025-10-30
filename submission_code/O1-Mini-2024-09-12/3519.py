from typing import List
from collections import defaultdict

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        counts = [defaultdict(int) for _ in range(n)]
        for x, y in pick:
            counts[x][y] += 1
        winning = 0
        for i in range(n):
            if any(count >= i + 1 for count in counts[i].values()):
                winning += 1
        return winning