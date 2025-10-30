class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        from collections import defaultdict, Counter
        
        # Count picks for each player
        player_picks = defaultdict(Counter)
        for player, color in pick:
            player_picks[player][color] += 1
        
        winners = 0
        for player in range(n):
            if player in player_picks:
                max_same_color = max(player_picks[player].values())
                if max_same_color > player:
                    winners += 1
        
        return winners