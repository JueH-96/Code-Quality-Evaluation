from typing import List
from collections import defaultdict

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # Initialize a list of dictionaries to track each player's color counts
        player_counts = [defaultdict(int) for _ in range(n)]
        
        # Populate the counts based on the picks
        for x, y in pick:
            player_counts[x][y] += 1
        
        result = 0
        
        # Check each player's condition
        for i in range(n):
            counts = player_counts[i]
            current_max = 0
            if counts:  # if there are any picks for this player
                current_max = max(counts.values())
            if current_max >= (i + 1):
                result += 1
        
        return result