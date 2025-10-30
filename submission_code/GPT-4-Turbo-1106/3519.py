class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        from collections import defaultdict
        
        # Create a dictionary to store the count of each color picked by each player
        player_picks = defaultdict(lambda: defaultdict(int))
        
        # Count the number of times each player picks each color
        for player, color in pick:
            player_picks[player][color] += 1
        
        # Initialize the count of winners
        winners = 0
        
        # Check for each player if they win
        for player in range(n):
            for color in player_picks[player]:
                if player_picks[player][color] > player:
                    winners += 1
                    break  # Once a player wins, no need to check other colors
        
        return winners