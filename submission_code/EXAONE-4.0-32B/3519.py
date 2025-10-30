from typing import List

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        count = [[0] * 11 for _ in range(n)]
        
        for player, color in pick:
            if player < n:
                count[player][color] += 1
                
        winners = 0
        for i in range(n):
            if any(count[i][c] >= i + 1 for c in range(11)):
                winners += 1
                
        return winners