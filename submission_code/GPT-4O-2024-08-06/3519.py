class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        from collections import defaultdict
        
        # Dictionary to store the count of each color picked by each player
        player_color_count = defaultdict(lambda: defaultdict(int))
        
        # Fill the dictionary with the number of balls of each color picked by each player
        for player, color in pick:
            player_color_count[player][color] += 1
        
        # Count the number of players who win
        winning_players = 0
        
        for player in range(n):
            # Check if the player has picked more than 'player' balls of any color
            for color, count in player_color_count[player].items():
                if count > player:
                    winning_players += 1
                    break  # No need to check other colors for this player
        
        return winning_players