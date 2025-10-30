import collections # collections can be useful for alternative implementations, but not strictly needed here
from typing import List # Required for type hints

class Solution:
    """
    Implements the solution to find the count of winning players in a ball picking game.
    A player 'i' wins if they pick strictly more than 'i' balls of the same color.
    """
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        """
        Calculates the number of players who win the game based on the provided picks.

        Args:
            n: The total number of players, indexed from 0 to n-1.
            pick: A list of lists, where each inner list [player_idx, color_idx] 
                  represents a pick by a player.

        Returns:
            An integer representing the count of players who meet the winning condition.
        """
        
        # Define the maximum possible color index based on constraints.
        # Colors are 0-indexed and go up to 10.
        max_color = 10 
        
        # Initialize a 2D list (or matrix) to store the counts of balls per color for each player.
        # `counts[player][color]` will store the number of balls of `color` picked by `player`.
        # The dimensions are n rows (for players 0 to n-1) and (max_color + 1) columns (for colors 0 to 10).
        # Initialize all counts to zero.
        counts = [[0] * (max_color + 1) for _ in range(n)]

        # Iterate through the list of picks to populate the counts matrix.
        for player, color in pick:
            # Constraints: 0 <= player < n and 0 <= color <= 10. These indices are guaranteed to be valid.
            # Increment the count for the specific player and color combination.
            counts[player][color] += 1

        # Initialize a counter for the number of winning players.
        winning_players_count = 0

        # Iterate through each player, indexed from i = 0 to n-1, to check if they win.
        for i in range(n):
            # Assume the player hasn't won yet for this iteration.
            player_wins = False
            
            # Check the counts for each possible color for the current player 'i'.
            # Colors range from 0 to max_color (inclusive).
            for color in range(max_color + 1):
                # The winning condition for player 'i' is picking strictly more than 'i' balls
                # of any single color. Check if the count for the current color exceeds 'i'.
                # `counts[i][color]` holds the number of balls of `color` picked by player `i`.
                if counts[i][color] > i:
                    # If the condition is met for this color, the player 'i' wins.
                    player_wins = True
                    # We found a color that makes player 'i' win, so we don't need to check other colors for this player.
                    # We can break the inner loop.
                    break 
            
            # If the flag `player_wins` is True, it means the player met the winning condition for at least one color.
            if player_wins:
                # Increment the total count of winning players.
                winning_players_count += 1

        # After checking all players, return the total count of winners.
        return winning_players_count