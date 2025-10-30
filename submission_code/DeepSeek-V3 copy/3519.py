from collections import defaultdict

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # Initialize a dictionary to count the number of balls each player has for each color
        player_color_count = defaultdict(lambda: defaultdict(int))
        
        # Populate the dictionary with the picks
        for p in pick:
            player, color = p
            player_color_count[player][color] += 1
        
        # Initialize the count of winning players
        winning_count = 0
        
        # Check each player to see if they win
        for player in range(n):
            # Get the counts of all colors for the current player
            color_counts = player_color_count[player].values()
            # Check if any color count is greater than the player's index + 1
            if any(count > (player + 1) for count in color_counts):
                winning_count += 1
        
        return winning_count