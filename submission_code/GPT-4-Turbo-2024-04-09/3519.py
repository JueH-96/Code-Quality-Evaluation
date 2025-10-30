class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        from collections import defaultdict
        
        # Dictionary to hold the count of each color picked by each player
        player_picks = defaultdict(lambda: defaultdict(int))
        
        # Fill the dictionary with the picks data
        for player, color in pick:
            player_picks[player][color] += 1
        
        # Count the number of winners
        winners = 0
        
        # Check each player if they win
        for player in range(n):
            # Get the maximum number of balls of the same color picked by this player
            max_picks = max(player_picks[player].values(), default=0)
            # Check if this player wins
            if max_picks > player:
                winners += 1
        
        return winners