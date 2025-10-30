from collections import defaultdict
from typing import List

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        player_balls = defaultdict(lambda: defaultdict(int))
        
        for player, color in pick:
            player_balls[player][color] += 1
        
        win_count = 0
        for player in range(n):
            max_balls_of_same_color = max(player_balls[player].values(), default=0)
            if max_balls_of_same_color > player:
                win_count += 1
        
        return win_count