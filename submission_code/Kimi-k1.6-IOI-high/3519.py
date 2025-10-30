from collections import defaultdict
from typing import List

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        color_counts = [defaultdict(int) for _ in range(n)]
        for x, y in pick:
            color_counts[x][y] += 1
        
        count = 0
        for i in range(n):
            for cnt in color_counts[i].values():
                if cnt > i:
                    count += 1
                    break  # No need to check other colors once one is found
        return count