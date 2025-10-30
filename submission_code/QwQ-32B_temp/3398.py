from typing import List

class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # Check each possible 2x2 square in the 3x3 grid
        for i in range(2):
            for j in range(2):
                # Count the number of 'B's in the current 2x2 square
                count_b = 0
                for dx in 0, 1:
                    for dy in 0, 1:
                        if grid[i + dx][j + dy] == 'B':
                            count_b += 1
                # Check if the count allows forming a uniform square with at most one flip
                if count_b in {0, 1, 3, 4}:
                    return True
        return False