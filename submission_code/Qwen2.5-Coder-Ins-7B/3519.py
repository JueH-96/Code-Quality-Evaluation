class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        player_wins = 0
        player_ball_count = [0] * n
        
        for player, ball in pick:
            player_ball_count[player] += 1
            if player_ball_count[player] > player:
                player_wins += 1
        
        return player_wins