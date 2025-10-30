class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # Iterate over all possible top-left corners of 2x2 squares
        for i in range(2):
            for j in range(2):
                b_count = 0
                # Check each cell in the 2x2 square starting at (i, j)
                for dx in 0, 1:
                    for dy in 0, 1:
                        if grid[i + dx][j + dy] == 'B':
                            b_count += 1
                # Calculate W count based on B count
                w_count = 4 - b_count
                # If either count is 3 or 4, return True
                if b_count >= 3 or w_count >= 3:
                    return True
        # No valid square found after checking all possibilities
        return False