from typing import List

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        """
        Calculates the number of winning players in a ball-picking game.

        Args:
            n: The number of players in the game, numbered 0 to n-1.
            pick: A 2D array where pick[i] = [x_i, y_i] represents that
                  player x_i picked a ball of color y_i.

        Returns:
            The number of players who win the game. A player i wins if they
            pick strictly more than i balls of the same color.
        """
        
        # Use a 2D list to store the count of balls for each player and color.
        # Dimensions: n rows for players (0 to n-1), 11 columns for colors (0 to 10).
        # This is efficient due to the small constraints on n and color values.
        player_color_counts = [[0] * 11 for _ in range(n)]

        # Tally the number of balls of each color for each player from the pick data.
        for player, color in pick:
            player_color_counts[player][color] += 1
            
        # A player `i` wins if they pick strictly more than `i` balls of the same color.
        # This is equivalent to checking if the maximum count of balls of any single
        # color for player `i` is greater than `i`.
        
        # We iterate through each player, check their condition, and sum up the winners.
        # A generator expression is a concise and efficient way to express this.
        return sum(1 for player_id in range(n) if max(player_color_counts[player_id]) > player_id)