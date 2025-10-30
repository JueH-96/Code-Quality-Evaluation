class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        squares = [
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(0, 1), (0, 2), (1, 1), (1, 2)],
            [(1, 0), (1, 1), (2, 0), (2, 1)],
            [(1, 1), (1, 2), (2, 1), (2, 2)]
        ]
        
        for square in squares:
            countB = 0
            countW = 0
            for i, j in square:
                if grid[i][j] == 'B':
                    countB += 1
                else:
                    countW += 1
            if countB >= 3 or countW >= 3:
                return True
        return False