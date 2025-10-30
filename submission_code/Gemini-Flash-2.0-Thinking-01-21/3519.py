class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        win_count = 0
        for player_id in range(n):
            color_counts = {}
            for p in pick:
                if p[0] == player_id:
                    color = p[1]
                    color_counts[color] = color_counts.get(color, 0) + 1
            max_count = 0
            for color in color_counts:
                max_count = max(max_count, color_counts[color])
            if max_count > player_id:
                win_count += 1
        return win_count