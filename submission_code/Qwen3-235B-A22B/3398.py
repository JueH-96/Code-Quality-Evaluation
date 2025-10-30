class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # Iterate through all possible 2x2 squares in the grid
        for i in range(2):
            for j in range(2):
                # Count 'B's and 'W's in the current 2x2 square
                b_count = 0
                w_count = 0
                for dx in range(2):
                    for dy in range(2):
                        cell = grid[i + dx][j + dy]
                        if cell == 'B':
                            b_count += 1
                        else:
                            w_count += 1
                # If any of the color counts are 3 or more, return True
                if b_count >= 3 or w_count >= 3:
                    return True
        # No suitable square found after checking all possibilities
        return False