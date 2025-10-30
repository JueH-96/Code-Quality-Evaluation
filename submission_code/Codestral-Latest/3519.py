class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        from collections import defaultdict

        # Dictionary to count the number of balls picked by each player for each color
        color_count = defaultdict(lambda: defaultdict(int))

        # Populate the color_count dictionary
        for player, color in pick:
            color_count[player][color] += 1

        # Count the number of winning players
        winning_players = 0
        for player in range(n):
            for color in color_count[player]:
                if color_count[player][color] > player:
                    winning_players += 1
                    break

        return winning_players