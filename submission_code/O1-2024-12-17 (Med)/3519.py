class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        from collections import defaultdict
        
        # Keep track of how many balls of each color each player picks
        color_counts = [defaultdict(int) for _ in range(n)]
        
        for player, color in pick:
            color_counts[player][color] += 1
        
        winners = 0
        # Check for each player if they meet their winning condition: 
        # at least (i+1) balls of the same color.
        for i in range(n):
            required = i + 1
            has_won = any(count >= required for count in color_counts[i].values())
            if has_won:
                winners += 1
        
        return winners