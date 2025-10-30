class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # Create a dictionary to store the count of balls picked by each player for each color
        player_balls = [{i: [0] * 11 for i in range(n)} for _ in range(n)]
        
        # Count the number of balls picked by each player for each color
        for player, color in pick:
            player_balls[player][color][color] += 1
        
        # Determine the number of winning players
        winning_players = 0
        for player in range(n):
            if any(count > player for count in player_balls[player][player]):
                winning_players += 1
        
        return winning_players