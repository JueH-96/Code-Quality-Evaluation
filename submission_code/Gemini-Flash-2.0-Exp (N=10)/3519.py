class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        player_picks = [{} for _ in range(n)]
        for x, y in pick:
            if y not in player_picks[x]:
                player_picks[x][y] = 0
            player_picks[x][y] += 1
        
        win_count = 0
        for i in range(n):
            wins = False
            for color, count in player_picks[i].items():
                if count > i:
                    wins = True
                    break
            if wins:
                win_count += 1
        return win_count