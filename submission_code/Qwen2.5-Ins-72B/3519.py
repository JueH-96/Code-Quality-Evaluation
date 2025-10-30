from collections import defaultdict
from typing import List

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # Create a dictionary to store the count of balls picked by each player for each color
        player_picks = defaultdict(lambda: defaultdict(int))
        
        # Populate the dictionary with the given picks
        for x, y in pick:
            player_picks[x][y] += 1
        
        # Count the number of players who win the game
        winners = 0
        for player in range(n):
            for color, count in player_picks[player].items():
                if count > player:
                    winners += 1
                    break
        
        return winners