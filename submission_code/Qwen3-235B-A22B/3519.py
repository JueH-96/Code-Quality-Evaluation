from collections import defaultdict
from typing import List

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # Initialize a list of defaultdicts to keep track of each player's ball counts per color
        players = [defaultdict(int) for _ in range(n)]
        
        # Populate the counts based on the pick list
        for x, y in pick:
            players[x][y] += 1
        
        # Count how many players meet the winning condition
        result = 0
        for x in range(n):
            counts = players[x]
            max_count = max(counts.values(), default=0)
            if max_count >= x + 1:
                result += 1
        
        return result