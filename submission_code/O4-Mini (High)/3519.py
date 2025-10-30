from typing import List
from collections import defaultdict

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # freq[i] will map color -> count of balls picked by player i
        freq = [defaultdict(int) for _ in range(n)]
        
        # Tally picks
        for player, color in pick:
            freq[player][color] += 1
        
        # Count how many players win
        winners = 0
        for i in range(n):
            if not freq[i]:
                continue
            # max number of same-colored balls picked by player i
            max_picks = max(freq[i].values())
            # player i wins if max_picks > i
            if max_picks > i:
                winners += 1
        
        return winners