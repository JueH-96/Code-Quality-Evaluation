class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        player_picks = {}
        for player, color in pick:
            if player not in player_picks:
                player_picks[player] = {}
            player_picks[player][color] = player_picks[player].get(color, 0) + 1
        
        winners = 0
        for player in range(n):
            if player in player_picks:
                max_picks = max(player_picks[player].values())
                if max_picks > player:
                    winners += 1
            elif player == 0:
                winners += 1
        
        return winners