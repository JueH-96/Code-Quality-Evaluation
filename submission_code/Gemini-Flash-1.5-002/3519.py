class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        player_picks = {}
        for p, c in pick:
            if p not in player_picks:
                player_picks[p] = {}
            if c not in player_picks[p]:
                player_picks[p][c] = 0
            player_picks[p][c] += 1
        
        win_count = 0
        for i in range(n):
            if i not in player_picks:
                continue
            wins = False
            for c in player_picks[i]:
                if player_picks[i][c] > i:
                    wins = True
                    break
            if wins:
                win_count += 1
        return win_count