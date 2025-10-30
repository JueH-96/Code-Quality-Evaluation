from typing import List
from collections import defaultdict

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # Dictionary to store the count of each color picked by each player
        player_color_count = defaultdict(lambda: defaultdict(int))

        # Populate the dictionary with the given picks
        for x, y in pick:
            player_color_count[x][y] += 1

        # Count the number of players who win the game
        winning_count = 0
        for player in range(n):
            if any(count > player for count in player_color_count[player].values()):
                winning_count += 1

        return winning_count