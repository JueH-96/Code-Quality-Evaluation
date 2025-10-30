from collections import defaultdict
from typing import List

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # Initialize a list of defaultdicts to track each player's color counts
        counts = [defaultdict(int) for _ in range(n)]
        
        # Update the counts based on the pick array
        for x, y in pick:
            counts[x][y] += 1
        
        # Check each player to see if they meet the winning condition
        result = 0
        for i in range(n):
            # If any color count for player i exceeds i, increment the result
            if any(cnt > i for cnt in counts[i].values()):
                result += 1
        
        return result