class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        win_count = 0
        for player_index in range(n):
            color_counts = {}
            for p in pick:
                player = p[0]
                color = p[1]
                if player == player_index:
                    color_counts[color] = color_counts.get(color, 0) + 1
            is_winner = False
            for color in color_counts:
                if color_counts[color] > player_index:
                    is_winner = True
                    break
            if is_winner:
                win_count += 1
        return win_count