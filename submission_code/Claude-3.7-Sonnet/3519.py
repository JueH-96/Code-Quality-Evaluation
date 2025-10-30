class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        from collections import defaultdict
        
        # Dictionary to store count of each color for each player
        player_colors = defaultdict(lambda: defaultdict(int))
        
        # Count how many balls of each color each player picks
        for player, color in pick:
            player_colors[player][color] += 1
        
        # Count winning players
        winners = 0
        for player in range(n):
            # For player i to win, they need to pick strictly more than i balls of the same color
            max_balls = 0
            if player in player_colors:
                max_balls = max(player_colors[player].values())
            
            if max_balls > player:
                winners += 1
        
        return winners