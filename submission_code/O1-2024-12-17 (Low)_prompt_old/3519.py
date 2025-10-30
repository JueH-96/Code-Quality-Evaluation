class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        from collections import defaultdict
        
        # Dictionary to store color counts for each player
        # color_counts[player][color] = number of times player picks this color
        color_counts = [defaultdict(int) for _ in range(n)]
        
        # Populate color counts
        for x, y in pick:
            color_counts[x][y] += 1
        
        # Count how many players have at least (i+1) picks of the same color
        wins = 0
        for i in range(n):
            if any(count >= i + 1 for count in color_counts[i].values()):
                wins += 1
                
        return wins