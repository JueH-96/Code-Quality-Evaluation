class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # Initialize a list of dictionaries to track each player's color counts
        player_counts = [{} for _ in range(n)]
        for x, y in pick:
            if y in player_counts[x]:
                player_counts[x][y] += 1
            else:
                player_counts[x][y] = 1
        
        result = 0
        # Check each player's maximum count of any color
        for i in range(n):
            counts = player_counts[i]
            max_count = max(counts.values()) if counts else 0
            if max_count > i:
                result += 1
        
        return result