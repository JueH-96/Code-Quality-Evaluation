from collections import defaultdict
from typing import List

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # Step 1: Initialize a data structure to store counts of balls picked by each player.
        # player_counts is a list where each element corresponds to a player.
        # player_counts[player_id] will be a defaultdict(int)
        # This defaultdict will store how many balls of each color 'color'
        # player 'player_id' has picked (color -> count).
        # Using defaultdict(int) automatically initializes new color counts to 0.
        player_counts = [defaultdict(int) for _ in range(n)]

        # Step 2: Populate the player_counts data structure.
        # Iterate through each pick event in the 'pick' array.
        for player_id, color in pick:
            # Increment the count for the specific color picked by the current player.
            # Example: If pick is [0, 0], player_counts[0][0] becomes 1.
            # If another [0, 0] comes, player_counts[0][0] becomes 2.
            player_counts[player_id][color] += 1

        # Step 3: Initialize a counter for the number of players who win the game.
        winning_players_count = 0

        # Step 4: Iterate through each player to determine if they win.
        # Players are indexed from 0 to n-1.
        for player_id in range(n):
            # According to the problem statement:
            # Player i wins if they pick strictly more than i balls of the same color.
            # This means:
            # Player 0 wins if they pick at least 0 + 1 = 1 ball of the same color.
            # Player 1 wins if they pick at least 1 + 1 = 2 balls of the same color.
            # Player i wins if they pick at least i + 1 balls of the same color.
            # So, the winning threshold for player 'player_id' is 'player_id + 1'.
            threshold = player_id + 1
            
            # Get the dictionary of color counts specifically for the current player.
            counts_for_current_player = player_counts[player_id]
            
            # Iterate through the colors and their respective counts that this player has picked.
            # A player wins as soon as they meet the condition for *any* single color.
            for color_picked, count in counts_for_current_player.items():
                if count >= threshold:
                    # If the count of a specific color meets or exceeds the required threshold,
                    # this player wins.
                    winning_players_count += 1
                    # Since a player only needs to win based on one color,
                    # we can stop checking other colors for this player and move to the next.
                    break
            
        # Step 5: Return the total count of players who won the game.
        return winning_players_count