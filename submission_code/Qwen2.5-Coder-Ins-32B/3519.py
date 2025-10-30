from typing import List
from collections import defaultdict

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # Dictionary to count the number of balls each player picks of each color
        player_color_count = defaultdict(lambda: defaultdict(int))
        
        # Count the balls picked by each player for each color
        for player, color in pick:
            player_color_count[player][color] += 1
        
        # Set to keep track of winning players
        winning_players = set()
        
        # Check each player's ball counts
        for player in range(n):
            for color, count in player_color_count[player].items():
                if count >= player + 1:
                    winning_players.add(player)
                    break
        
        return len(winning_players)