class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        # Initialize a 2D list to count the number of times each color is picked by each player
        count_matrix = [[0] * 11 for _ in range(n)]
        
        # Count the picks for each player and color
        for player, color in pick:
            count_matrix[player][color] += 1
        
        # Count the number of winning players
        win_count = 0
        for i in range(n):
            # Find the maximum count of any color for player i
            max_count = max(count_matrix[i])
            # Check if the max count is at least i + 1
            if max_count >= i + 1:
                win_count += 1
        
        return win_count