from typing import List

class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        def check_square(x, y, color):
            count = 0
            for i in range(x, x + 2):
                for j in range(y, y + 2):
                    if grid[i][j] != color:
                        count += 1
            return count <= 1
        
        for i in range(2):
            for j in range(2):
                if check_square(i, j, 'B') or check_square(i, j, 'W'):
                    return True
        return False