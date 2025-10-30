class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # Initialize a dictionary to store the count of each color picked by each player
        player_color_count = {i: {} for i in range(n)}
        
        # Count the number of times each player picks each color
        for player, color in pick:
            if color in player_color_count[player]:
                player_color_count[player][color] += 1
            else:
                player_color_count[player][color] = 1
        
        # Count the number of players who win the game
        winning_players = 0
        for player, color_count in player_color_count.items():
            for color, count in color_count.items():
                if count > player:
                    winning_players += 1
                    break
        
        return winning_players