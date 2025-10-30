from typing import List
from collections import defaultdict

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # Initialize a dictionary to hold player picks: {player: {color: count}}
        player_picks = defaultdict(lambda: defaultdict(int))
        
        # Populate the player_picks dictionary
        for x, y in pick:
            player_picks[x][y] += 1
        
        # Count the number of winning players
        winners = 0
        for i in range(n):
            if not player_picks[i]:
                max_count = 0
            else:
                max_count = max(player_picks[i].values())
            if max_count >= i + 1:
                winners += 1
                
        return winners