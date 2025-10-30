class Solution:
    def winningPlayerCount(self, n: int, pick: [[int]]) -> int:
        player_picks = [{} for _ in range(n)]
        for p, c in pick:
            if c in player_picks[p]:
                player_picks[p][c] += 1
            else:
                player_picks[p][c] = 1
        winners = 0
        for i in range(n):
            for c in player_picks[i]:
                if player_picks[i][c] > i:
                    winners += 1
                    break
        return winners