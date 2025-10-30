from collections import defaultdict
from typing import List

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # Initialize a list of defaultdicts to track color counts for each player
        counts = [defaultdict(int) for _ in range(n)]
        for x, y in pick:
            counts[x][y] += 1
        
        res = 0
        for i in range(n):
            required = i + 1
            # Check each color's count for the current player
            for cnt in counts[i].values():
                if cnt >= required:
                    res += 1
                    break  # No need to check other colors once a qualifying one is found
        return res