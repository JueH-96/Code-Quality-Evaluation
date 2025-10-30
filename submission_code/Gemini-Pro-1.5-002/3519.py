class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        counts = [{} for _ in range(n)]
        for player, color in pick:
            counts[player][color] = counts[player].get(color, 0) + 1

        wins = 0
        for i in range(n):
            won = False
            for count in counts[i].values():
                if count >= i + 1:
                    won = True
                    break
            if won:
                wins += 1
        return wins