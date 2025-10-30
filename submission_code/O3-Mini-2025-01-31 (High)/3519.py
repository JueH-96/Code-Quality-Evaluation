from typing import List
from collections import defaultdict

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # Create a list of dictionaries to count the number of picks for each color per player.
        players = [defaultdict(int) for _ in range(n)]
        
        # Process each pick entry.
        for player, color in pick:
            players[player][color] += 1
        
        win_count = 0
        # Check for each player if any color count meets or exceeds the winning threshold (i+1)
        for i in range(n):
            for count in players[i].values():
                if count >= i + 1:
                    win_count += 1
                    break
        return win_count