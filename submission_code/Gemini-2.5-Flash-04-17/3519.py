from collections import defaultdict
from typing import List

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        """
        Calculates the number of players who win the game based on their picks.

        Args:
            n: The total number of players (indexed 0 to n-1).
            pick: A list of picks, where pick[i] = [player_id, color].

        Returns:
            The number of players who win the game.
        """
        # Use a dictionary of dictionaries to store counts:
        # player_color_counts[player_id][color] = number of times player_id picked color
        # defaultdict simplifies handling cases where a player or color hasn't been seen yet.
        player_color_counts = defaultdict(lambda: defaultdict(int))

        # Populate the counts from the pick list
        # Constraints guarantee 0 <= player_id <= n-1 and 0 <= color <= 10
        for player_id, color in pick:
            player_color_counts[player_id][color] += 1

        winning_players_count = 0

        # Check each player from 0 to n-1
        for player_id in range(n):
            # Condition for player player_id to win:
            # They must pick strictly more than player_id balls of the same color.
            # This is equivalent to picking at least player_id + 1 balls of the same color.
            required_count = player_id + 1

            # A player wins if there is at least one color for which their count
            # meets or exceeds the required threshold.
            
            player_wins = False
            # Iterate through the counts of balls picked by this player for each color.
            # player_color_counts[player_id].values() returns an empty list if the player
            # is not in the counts dictionary or has picked 0 balls, which is handled correctly.
            for count in player_color_counts[player_id].values():
                if count >= required_count:
                    player_wins = True
                    break # Found a winning condition for this player, no need to check other colors

            if player_wins:
                winning_players_count += 1

        return winning_players_count