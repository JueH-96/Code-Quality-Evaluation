from typing import List
from collections import defaultdict

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # counts[player][color] = how many times 'player' picked 'color'
        counts = [defaultdict(int) for _ in range(n)]
        
        # Accumulate the picks
        for player, color in pick:
            # player is guaranteed to be in range 0..n-1 by constraints
            counts[player][color] += 1
        
        winners = 0
        
        # For each player, check if any color count exceeds the player's index
        for player in range(n):
            needed = player + 1          # strictly more than player index  ->  i + 1
            if any(c >= needed for c in counts[player].values()):
                winners += 1
        
        return winners