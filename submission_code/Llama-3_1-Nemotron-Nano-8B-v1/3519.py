from typing import List
from collections import defaultdict, Counter

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        player_colors = defaultdict(list)
        for x, y in pick:
            player_colors[x].append(y)
        win_count = 0
        for player in range(n):
            colors = player_colors[player]
            if not colors:
                continue
            cnt = Counter(colors)
            max_count = max(cnt.values())
            if max_count >= player + 1:
                win_count += 1
        return win_count