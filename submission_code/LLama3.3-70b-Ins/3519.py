from typing import List
from collections import defaultdict

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # Create a dictionary to store the count of each color for each player
        player_colors = defaultdict(lambda: defaultdict(int))
        
        # Count the colors for each player
        for player, color in pick:
            player_colors[player][color] += 1
        
        # Initialize the count of winning players
        winning_players = 0
        
        # Check each player
        for player in range(n):
            # Get the maximum count of colors for the current player
            max_color_count = max(player_colors[player].values(), default=0)
            
            # Check if the player wins
            if max_color_count > player:
                winning_players += 1
        
        return winning_players