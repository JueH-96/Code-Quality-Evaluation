class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        player_counts = [{} for _ in range(n)]
        for x_i, y_i in pick:
            if y_i in player_counts[x_i]:
                player_counts[x_i][y_i] +=1
            else:
                player_counts[x_i][y_i] =1
        win_count = 0
        for i in range(n):
            max_color_picked = max(player_counts[i].values(), default=0)
            if max_color_picked >= i+1:
                win_count +=1
        return win_count