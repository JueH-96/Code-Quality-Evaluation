from collections import defaultdict

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # Initialize a dictionary to count the number of balls each player has for each color
        player_color_counts = defaultdict(lambda: defaultdict(int))
        
        # Iterate through the pick list and update the counts
        for p in pick:
            player = p[0]
            color = p[1]
            player_color_counts[player][color] += 1
        
        # Initialize a set to keep track of winning players
        winning_players = set()
        
        # Iterate through each player and check if they win
        for player in range(n):
            # Get the counts of balls for each color for the current player
            color_counts = player_color_counts[player]
            # Check if any color count is greater than the player's index
            for color, count in color_counts.items():
                if count > player:
                    winning_players.add(player)
                    break  # No need to check other colors for this player
        
        # Return the number of unique winning players
        return len(winning_players)