from typing import List

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        counts = [[0] * 11 for _ in range(n)]
        
        for player, color in pick:
            counts[player][color] += 1
        
        wins = 0
        for i in range(n):
            if max(counts[i]) >= i + 1:
                wins += 1
        
        return wins