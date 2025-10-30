class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # Create dictionary to store color counts for each player
        player_colors = {}
        
        # Count colors picked by each player
        for p, c in pick:
            if p not in player_colors:
                player_colors[p] = {}
            if c not in player_colors[p]:
                player_colors[p][c] = 0
            player_colors[p][c] += 1
        
        winners = 0
        # Check each player
        for i in range(n):
            # Player i needs i+1 balls of same color to win
            if i in player_colors:
                # Get max count of any color for this player
                max_color_count = max(player_colors[i].values())
                # Player wins if they have more balls of same color than their index
                if max_color_count > i:
                    winners += 1
                    
        return winners