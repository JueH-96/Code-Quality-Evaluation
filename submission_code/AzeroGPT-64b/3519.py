from collections import defaultdict

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # Store the number of times a player picks a ball of a certain color.
        player_col_balls = defaultdict(lambda: defaultdict(int))
        for person, col in pick:
            player_col_balls[person][col] += 1

        # For each player, determine if they've won.
        winners = 0
        for i in range(n):
            if any(count >= i + 1 for count in player_col_balls[i].values()):
                winners += 1

        return winners