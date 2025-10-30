from typing import List
from collections import Counter, defaultdict

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # Count how many balls of each color each player picks
        player_color_counts = defaultdict(Counter)
        for player, color in pick:
            player_color_counts[player][color] += 1

        winners = 0
        # Check for each player if they picked > i balls of any one color
        for i in range(n):
            # The threshold for player i is i+1 picks of the same color
            needed = i + 1
            # If player i never picked anything, they can't win unless i == 0 and they have 0 picks (but 0 needs 1 pick)
            # So just check their color counts
            color_counts = player_color_counts.get(i, {})
            if any(count >= needed for count in color_counts.values()):
                winners += 1

        return winners