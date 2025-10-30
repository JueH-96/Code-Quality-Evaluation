from typing import List

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        player_colors = [{} for _ in range(n)]
        for player, color in pick:
            if color not in player_colors[player]:
                player_colors[player][color] = 0
            player_colors[player][color] += 1
        
        winners = 0
        for i in range(n):
            won = False
            for count in player_colors[i].values():
                if count > i:
                    won = True
                    break
            if won:
                winners += 1
        
        return winners