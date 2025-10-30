from typing import List
from collections import defaultdict

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # Dictionary to store the count of each color picked by each player
        player_picks = defaultdict(lambda: defaultdict(int))
        
        # Populate the dictionary with the pick data
        for player, color in pick:
            player_picks[player][color] += 1
        
        # Count the number of players who win the game
        winning_count = 0
        for player in range(n):
            # Check if the player has picked more than player+1 balls of any color
            for color, count in player_picks[player].items():
                if count > player:
                    winning_count += 1
                    break
        
        return winning_count