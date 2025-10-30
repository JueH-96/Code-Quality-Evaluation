class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # Create a dictionary to track balls picked by each player
        # player_picks[player][color] = count
        player_picks = {}
        
        # Count balls for each player and color
        for player, color in pick:
            if player not in player_picks:
                player_picks[player] = {}
            if color not in player_picks[player]:
                player_picks[player][color] = 0
            player_picks[player][color] += 1
        
        # Count winning players
        winners = 0
        
        # Check each player (0 to n-1)
        for player in range(n):
            if player in player_picks:
                # Check if player has any color with more than player number of balls
                max_balls_of_same_color = max(player_picks[player].values())
                if max_balls_of_same_color > player:
                    winners += 1
        
        return winners