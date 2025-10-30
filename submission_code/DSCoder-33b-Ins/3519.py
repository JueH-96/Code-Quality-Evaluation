from typing import List
from collections import defaultdict

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        player_balls = defaultdict(lambda: defaultdict(int))
        for player, color in pick:
            player_balls[player][color] += 1

        winners = 0
        for player in range(n):
            for color, count in player_balls[player].items():
                if count >= player + 1:
                    winners += 1
                    break

        return winners