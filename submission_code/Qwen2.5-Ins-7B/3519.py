from collections import defaultdict

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        color_count = defaultdict(int)
        winning_players = 0
        
        for player, color in pick:
            color_count[color] += 1
            if color_count[color] > player:
                winning_players += 1
        
        return winning_players