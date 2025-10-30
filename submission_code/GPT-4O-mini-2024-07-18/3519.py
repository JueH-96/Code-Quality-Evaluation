from collections import defaultdict
from typing import List

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        color_count = defaultdict(lambda: defaultdict(int))
        
        # Count the number of balls picked by each player for each color
        for player, color in pick:
            color_count[player][color] += 1
        
        winners = 0
        
        # Determine how many players win
        for player in range(n):
            # Check if the player has picked enough balls of any color
            for count in color_count[player].values():
                if count > player:
                    winners += 1
                    break
        
        return winners