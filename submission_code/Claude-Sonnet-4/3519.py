class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # Create a dictionary to track each player's color counts
        player_colors = {}
        
        # Initialize each player's color count dictionary
        for i in range(n):
            player_colors[i] = {}
        
        # Count balls of each color for each player
        for player, color in pick:
            if color not in player_colors[player]:
                player_colors[player][color] = 0
            player_colors[player][color] += 1
        
        # Count winning players
        winners = 0
        for player in range(n):
            # Player i wins if they have at least (i+1) balls of any single color
            required_balls = player + 1
            
            # Check if this player has enough balls of any color
            for color_count in player_colors[player].values():
                if color_count >= required_balls:
                    winners += 1
                    break  # Player wins, no need to check other colors
        
        return winners