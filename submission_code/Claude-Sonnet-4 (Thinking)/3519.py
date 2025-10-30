class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # List of dictionaries to store color counts for each player
        player_colors = [{} for _ in range(n)]
        
        # Process picks
        for player, color in pick:
            player_colors[player][color] = player_colors[player].get(color, 0) + 1
        
        # Count winners
        winners = 0
        for i in range(n):
            required = i + 1
            max_same_color = max(player_colors[i].values()) if player_colors[i] else 0
            
            if max_same_color >= required:
                winners += 1
        
        return winners