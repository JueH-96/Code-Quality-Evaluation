from typing import List
from collections import defaultdict

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # Initialize a list of defaultdicts for each player
        player_picks = [defaultdict(int) for _ in range(n)]
        
        # Populate the picks
        for x, y in pick:
            player_picks[x][y] += 1
        
        # Count the number of players who win
        win_count = 0
        for i in range(n):
            if player_picks[i]:
                max_count = max(player_picks[i].values())
                if max_count > i:
                    win_count += 1
            # If player_picks[i] is empty, max_count is 0, which is not > i
        return win_count