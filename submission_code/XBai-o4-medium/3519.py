from collections import defaultdict
from typing import List

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # Initialize a list of dictionaries to track color counts for each player
        color_counts = [defaultdict(int) for _ in range(n)]
        
        # Process each pick to populate the color counts
        for x, y in pick:
            color_counts[x][y] += 1
        
        result = 0
        
        # Check each player's maximum color count against their index
        for i in range(n):
            if color_counts[i]:  # Check if the player has any picks
                max_count = max(color_counts[i].values())
            else:
                max_count = 0
            
            if max_count > i:
                result += 1
        
        return result