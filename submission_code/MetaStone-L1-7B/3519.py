from collections import defaultdict, Counter
from typing import List

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        player_picks = defaultdict(list)
        for x, y in pick:
            player_picks[x].append(y)
        
        result = 0
        for x in range(n):
            if x not in player_picks:
                continue
            color_counts = Counter(player_picks[x])
            for cnt in color_counts.values():
                if cnt > x:
                    result += 1
                    break
        return result