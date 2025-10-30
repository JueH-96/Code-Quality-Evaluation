from collections import defaultdict
from typing import List

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # Dictionary to store the count of balls picked by each player for each color
        player_balls = defaultdict(lambda: defaultdict(int))
        
        # Iterate through the pick list to populate the player_balls dictionary
        for player, color in pick:
            player_balls[player][color] += 1
        
        # Initialize the count of winning players to 0
        winning_count = 0
        
        # Iterate through the player_balls dictionary to check if a player wins
        for player, balls in player_balls.items():
            # Check if the player has picked enough balls of the same color to win
            if any(count > player for count in balls.values()):
                winning_count += 1
        
        # Return the count of winning players
        return winning_count